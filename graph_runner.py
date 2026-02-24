import tensorflow as tf
import numpy as np
import os
from tensorflow.core.framework import graph_pb2

model_path = "graph_def.pb"
batch_size = 100


def configure_musa_runtime():
    """
    Load libmusa_plugin.so if available and enable device placement logs.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.environ.get("MUSA_PLUGIN_PATH"),
        os.path.join(script_dir, "libmusa_plugin.so"),
        os.path.join(
            script_dir, "..", "tensorflow_musa_extension", "build", "libmusa_plugin.so"
        ),
        "/workspace/tensorflow_musa/build/libmusa_plugin.so",
    ]

    plugin_loaded = False
    for candidate in candidate_paths:
        if not candidate:
            continue
        plugin_path = os.path.abspath(candidate)
        if os.path.isfile(plugin_path):
            try:
                tf.load_library(plugin_path)
                print(f"[MUSA] plugin loaded: {plugin_path}")
                plugin_loaded = True
                break
            except Exception as e:
                print(f"[MUSA] failed to load plugin {plugin_path}: {e}")

    if not plugin_loaded:
        print("[MUSA] libmusa_plugin.so not found; run will continue without plugin.")
        print("[MUSA] set MUSA_PLUGIN_PATH=/abs/path/libmusa_plugin.so to force a path.")

    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(True)
    print("[MUSA] tf.config.set_soft_device_placement(True)")
    print("[MUSA] tf.debugging.set_log_device_placement(True)")


def infer_placeholder_shape_from_usage(graph_def, placeholder_name):
    """
    通过分析图中使用该 Placeholder 的节点来推断其形状
    """
    # 查找使用该 Placeholder 的节点
    for node in graph_def.node:
        for input_name in node.input:
            clean_input = input_name.split(":")[0].lstrip("^")
            if clean_input == placeholder_name:
                # 找到使用该 Placeholder 的节点
                if node.op == "MatMul" or node.op == "Tensordot":
                    # 对于矩阵乘法，尝试从输出形状推断
                    if "_output_shapes" in node.attr:
                        output_shapes = node.attr["_output_shapes"].list.shape
                        if len(output_shapes) > 0:
                            output_shape = output_shapes[0]
                            if len(output_shape.dim) == 2:
                                # 假设这是权重矩阵，通常是 [input_dim, output_dim]
                                input_dim = (
                                    output_shape.dim[1].size
                                    if output_shape.dim[1].size > 0
                                    else 32
                                )
                                output_dim = (
                                    output_shape.dim[0].size
                                    if output_shape.dim[0].size > 0
                                    else 32
                                )
                                # 对于权重矩阵，通常形状是 [input_features, output_features]
                                # 但具体的形状需要根据实际情况调整
                                return [
                                    input_dim * 24,
                                    output_dim,
                                ]  # 使用一个合理的倍数
                elif node.op == "BiasAdd":
                    # 对于偏置加法，偏置通常是一维的
                    if "_output_shapes" in node.attr:
                        output_shapes = node.attr["_output_shapes"].list.shape
                        if len(output_shapes) > 0:
                            output_shape = output_shapes[0]
                            if len(output_shape.dim) >= 1:
                                bias_size = (
                                    output_shape.dim[-1].size
                                    if output_shape.dim[-1].size > 0
                                    else 32
                                )
                                return [bias_size]

    # 如果无法推断，返回 None
    return None


def load_graph_and_get_placeholders(pb_path):
    """
    加载图并获取所有 placeholder 节点信息
    """
    print("=== 加载图并分析 Placeholder 节点 ===")

    # 读取 pb 文件
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())

    print(f"图加载成功，总节点数: {len(graph_def.node)}")

    # 查找所有 placeholder 节点
    placeholders = {}

    for node in graph_def.node:
        if node.op == "Placeholder":
            # 获取数据类型
            dtype_enum = node.attr["dtype"].type
            dtype_map = {
                tf.float32.as_datatype_enum: np.float32,
                tf.int32.as_datatype_enum: np.int32,
                tf.int64.as_datatype_enum: np.int64,
                tf.bool.as_datatype_enum: np.bool_,
                tf.string.as_datatype_enum: np.str_,
            }
            dtype = dtype_map.get(dtype_enum, np.float32)

            # 获取形状 - 优先检查 shape 属性，然后检查 _output_shapes 属性
            shape = []
            shape_found = False

            # 首先尝试从 shape 属性获取
            if "shape" in node.attr:
                shape_proto = node.attr["shape"].shape
                if not shape_proto.unknown_rank:
                    for dim in shape_proto.dim:
                        if dim.size == -1:
                            shape.append(None)  # 动态维度
                        else:
                            shape.append(dim.size)
                    shape_found = True
                    print(f"从 shape 属性获取形状: {shape}")

            # 如果 shape 属性没有有效信息，尝试从 _output_shapes 属性获取
            if not shape_found and "_output_shapes" in node.attr:
                output_shapes = node.attr["_output_shapes"].list.shape
                if len(output_shapes) > 0:
                    shape_proto = output_shapes[0]  # 取第一个输出形状
                    if not shape_proto.unknown_rank:
                        for dim in shape_proto.dim:
                            if dim.size == -1:
                                shape.append(None)  # 动态维度
                            else:
                                shape.append(dim.size)
                        shape_found = True
                        print(f"从 _output_shapes 属性获取形状: {shape}")

            # 如果仍然没有找到形状信息，尝试从使用情况推断
            if not shape_found:
                inferred_shape = infer_placeholder_shape_from_usage(
                    graph_def, node.name
                )
                if inferred_shape:
                    shape = inferred_shape
                    print(f"为 {node.name} 推断出形状: {shape}")
                else:
                    shape = []
                    print(f"无法获取 {node.name} 的形状信息，使用空形状")

            placeholders[node.name] = {"dtype": dtype, "shape": shape}

            print(f"Placeholder: {node.name}")
            print(f"  数据类型: {dtype}")
            print(f"  形状: {shape}")

    print(f"找到 {len(placeholders)} 个 Placeholder 节点")
    return graph_def, placeholders


def create_mock_data(placeholders, batch_size):
    """
    根据 placeholder 信息创建 mock 数据
    """
    print("\n=== 创建 Mock 数据 ===")

    feed_dict = {}

    for name, info in placeholders.items():
        shape = info["shape"]
        dtype = info["dtype"]

        # 处理形状，将 None 替换为 batch_size
        mock_shape = []
        for dim in shape:
            if dim is None:
                mock_shape.append(batch_size)
            elif dim == 0:
                # 维度为0的情况，保持为0
                mock_shape.append(0)
            else:
                mock_shape.append(dim)

        # 如果形状为空列表，说明是标量，但某些 Placeholder 可能需要特定形状
        # 检查名称中是否包含特定模式来推断正确的形状
        if not mock_shape:
            # 对于 ReadVariableOp/resource 类型的 Placeholder，尝试从名称推断形状
            if "/ReadVariableOp/resource" in name:
                # 这些通常是权重或偏置参数，需要根据上下文推断形状
                # 暂时使用一个合理的默认形状
                if "BiasAdd" in name:
                    # 偏置通常是一维向量
                    mock_shape = [32]  # 默认偏置大小
                elif "MatMul" in name or "Tensordot" in name:
                    # 权重矩阵通常是二维
                    mock_shape = [64, 32]  # 默认权重矩阵大小
                else:
                    # 其他情况使用标量
                    mock_shape = []
            else:
                # 其他标量 Placeholder 保持标量
                mock_shape = []

        # 生成 mock 数据
        if dtype == np.float32:
            # 生成随机浮点数据
            mock_data = np.random.normal(0.0, 1.0, mock_shape).astype(dtype)
        elif dtype == np.int32:
            # 生成随机整数数据
            mock_data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.int64:
            # 生成随机长整数数据
            mock_data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.bool_:
            # 生成随机布尔数据
            mock_data = np.random.choice([True, False], mock_shape).astype(dtype)
        else:
            # 默认生成浮点数据
            mock_data = np.random.normal(0.0, 1.0, mock_shape).astype(np.float32)

        feed_dict[name + ":0"] = mock_data

        print(f"Mock 数据 - {name}:")
        print(f"  形状: {mock_shape}")
        print(f"  数据类型: {dtype}")
        print(f"  数据范围: [{np.min(mock_data):.4f}, {np.max(mock_data):.4f}]")

    return feed_dict


def run_inference(graph_def, feed_dict, output_node_name="predicts"):
    """
    执行图推理
    """
    print(f"\n=== 执行图推理 ===")
    print(f"输出节点: {output_node_name}")

    # 创建新的图
    with tf.Graph().as_default() as graph:
        # 导入图定义
        tf.import_graph_def(graph_def, name="")

        # 获取输入和输出张量
        input_tensors = {}
        for name in feed_dict.keys():
            try:
                input_tensors[name] = graph.get_tensor_by_name(name)
            except KeyError:
                print(f"警告: 找不到输入张量 {name}")

        # 获取输出张量
        try:
            output_tensor = graph.get_tensor_by_name(f"{output_node_name}:0")
        except KeyError:
            print(f"错误: 找不到输出张量 {output_node_name}:0")
            return None

        # 创建会话并执行推理
        with tf.compat.v1.Session(graph=graph) as sess:
            try:
                # 准备 feed_dict
                session_feed_dict = {}
                for name, data in feed_dict.items():
                    if name in input_tensors:
                        session_feed_dict[input_tensors[name]] = data

                print(f"开始推理...")
                result = sess.run(output_tensor, feed_dict=session_feed_dict)

                print(f"推理成功!")
                print(f"输出形状: {result.shape}")
                print(f"输出数据类型: {result.dtype}")
                print(f"输出数据范围: [{np.min(result):.6f}, {np.max(result):.6f}]")
                print(f"输出数据统计:")
                print(f"  均值: {np.mean(result):.6f}")
                print(f"  标准差: {np.std(result):.6f}")

                # 显示部分输出数据
                if result.size <= 20:
                    print(f"完整输出数据:\n{result}")
                else:
                    print(f"输出数据前10个值:\n{result.flatten()[:10]}")

                return result

            except Exception as e:
                print(f"推理执行失败: {e}")
                import traceback

                traceback.print_exc()
                return None


def main():
    """
    主函数
    """
    print("=== TensorFlow 图推理脚本 ===")
    print(f"模型路径: {model_path}")
    print(f"批次大小: {batch_size}")
    print()

    try:
        # 1. 加载图并获取 placeholder 信息
        configure_musa_runtime()
        graph_def, placeholders = load_graph_and_get_placeholders(model_path)

        if not placeholders:
            print("错误: 没有找到任何 Placeholder 节点")
            return

        # 2. 创建 mock 数据
        feed_dict = create_mock_data(placeholders, batch_size)

        # 3. 执行推理
        result = run_inference(graph_def, feed_dict, "predicts")

        if result is not None:
            print("\n=== 推理完成 ===")
            print("脚本执行成功!")
        else:
            print("\n=== 推理失败 ===")
            print("脚本执行失败!")

    except Exception as e:
        print(f"脚本执行出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
