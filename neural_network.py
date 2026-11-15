import argparse
import numpy as np

import struct
import time
import os

#encoding map is used for binary output mode, it maps the 62 classes to a binary vector with a hamming distance of at least 3 between each class for easier training and error correction
ENCODING_MAP = [list((elem >> i) & 1 for i in range(10)) for elem in [7,25,30,42,45,51,52,75,76,82,85,97,102,120,127,385,390,408,415,427,428,434,437,458,461,467,468,480,487,505,510,642,645,667,668,680,687,689,694,713,718,720,727,739,740,762,765,771,772,794,797,809,814,816,823,840,847,849,854,866,869,891,892]]

def main():
    np.set_printoptions(linewidth=200)
    parser = argparse.ArgumentParser('NeuralNetwork')

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dest", type=str, default="neural_network.wgt")
    train_parser.add_argument("--data", type=str, default="train_data")
    train_parser.add_argument("--batch-size", type=int, default=100)
    train_parser.add_argument("--layers", type=int, nargs="+", default=[32, 24])
    train_parser.add_argument("--input-size", type=int, default=784)
    train_parser.add_argument("--output-size", type=int, default=10)
    train_parser.add_argument("--output-mode", choices=["class", "binary"], default="class")
    train_parser.add_argument("--learning-rate", type=float, default=0.1)
    train_parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh", "leaky-relu"], default="relu")
    train_parser.add_argument("--leaky-relu-coef", type=float, default=0.01)
    train_parser.add_argument("--garbage", default=0.00, type=float, help="Percentage of garbage data to add to the training set (0.0-1.0). For binary output mode, it will train the model to output a specific garbage class (62) for garbage data. For class output mode, it will train the model to output the most uniform class distribution possible for garbage data.")
    train_parser.add_argument("--steps", type=int, default=10000)
    train_parser.add_argument("--model", type=str)
    
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--model", type=str, default="neural_network.wgt")
    test_parser.add_argument("--data", type=str, default="test_data")
    test_parser.add_argument("--garbage", default=0.0, type=float, help="Percentage of garbage data to add to the test set (0.0-1.0). For binary output mode, it will require the model to output a specific garbage class (62) for garbage data. For class output mode, it will require the model to output the most uniform class distribution possible for garbage data.")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", type=str, default="neural_network.wgt")
    run_parser.add_argument("--input", type=str, default="input.dat")
    run_parser.add_argument("--folder", action="store_true")
    run_parser.add_argument("--full", action="store_true")

    """render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--model", type=str, default="neural_network.wgt")
    render_parser.add_argument("--input", type=str, default="input.dat")"""

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--data", type=str, default="test_data")
    extract_parser.add_argument("--out", type=str, default="extracted")
    extract_parser.add_argument("--name", type=str, default="sample")
    extract_parser.add_argument("--count", type=int, default=1)
    extract_parser.add_argument("--random", action="store_true")
    extract_parser.add_argument("--index", type=int, nargs="+")

    args = parser.parse_args()

    if args.command == "train" :
        if args.output_mode == "binary" and args.output_size != 10:
            print("Output size must be 10 for binary output mode.")
            return
        train(args)
    elif args.command == "test":
        test(args)
    elif args.command == "run":
        run(args)
    elif args.command == "extract":
        extract(args)
    return

def test(args):
    print("Loading data and model...")
    images = load_idx_normalized(f"{args.data}/images.idx")
    labels = load_idx(f"{args.data}/labels.idx")
    model = load_model(args.model)

    n = len(images)
    mode_name = "class" if model.output_mode == 0 else "binary"
    print(f"Loaded {n} test samples (output mode: {mode_name})\n")

    total_sse = 0.0
    correct = 0
    classes = np.unique(labels)
    class_total = {int(c): 0 for c in classes}
    class_correct = {int(c): 0 for c in classes}

    start = time.time()
    step = max(1, n // 10)  # progress update every 10%

    for i in range(n):
        label = int(labels[i])
        result = np.asarray(execute_model(model, images[i])).ravel()

        # Build the expected output vector and the predicted label
        if model.output_mode == 0:
            expected = np.zeros(len(result))
            expected[label] = 1.0
            predicted = int(np.argmax(result))
        else:
            expected = np.asarray(class_to_binary(label)).ravel()
            predicted = binary_to_class(result)

        total_sse += np.sum((result - expected) ** 2)

        class_total[label] += 1
        if predicted == label:
            correct += 1
            class_correct[label] += 1

        if (i + 1) % step == 0 or i + 1 == n:
            pct = 100 * (i + 1) / n
            print(f"  [{pct:5.1f}%] {i + 1}/{n} | running accuracy: "
                  f"{100 * correct / (i + 1):.2f}% | running SSE: {total_sse:.4f}")

    elapsed = time.time() - start

    print("\n========== TEST RESULTS ==========")
    print(f"Samples          : {n}")
    print(f"Correct          : {correct}")
    print(f"Accuracy         : {100 * correct / n:.2f}%")
    print(f"Total SSE        : {total_sse:.4f}")
    print(f"Mean SSE/sample  : {total_sse / n:.6f}")
    print(f"Time             : {elapsed:.2f}s ({1000 * elapsed / n:.2f} ms/sample)")

    print("\nPer-class accuracy:")
    for c in sorted(class_total):
        t = class_total[c]
        if t > 0:
            print(f"  class {c}: {100 * class_correct[c] / t:6.2f}%  ({class_correct[c]}/{t})")
    print("==================================")

    return total_sse

def train(args):

    images = load_idx_normalized(f"{args.data}/images.idx")
    labels = load_idx(f"{args.data}/labels.idx")
    model = None

    if args.model:
        model = load_model(args.model)
        print(f"Loaded existing model from {args.model}")
    else:
        model = Model(args.layers, args.input_size, args.output_size, ["class", "binary"].index(args.output_mode), ["relu", "sigmoid", "tanh", "leaky-relu"].index(args.activation), args.leaky_relu_coef)
        generate_model_weights(model)
        print(f"Created new model | layers: {[args.input_size] + args.layers + [args.output_size]} | output mode: {args.output_mode} | activation: {args.activation}")

    print(f"Training on {len(images)} samples | {args.steps} steps | batch size {args.batch_size}\n")
    start = time.time()
    cooldown = 0

    indices = np.random.permutation(len(images))
    batch_offset = 0

    for i in range(args.steps):

        step = i + 1

        # Shuffle dataset when the current shuffled set is done
        if batch_offset >= len(images):
            indices = np.random.permutation(len(images))
            batch_offset = 0

        batch_id = indices[np.arange(batch_offset, batch_offset + args.batch_size) % len(images)]
        batch_offset += args.batch_size

        gradient = np.zeros(args.output_size)
        batch_sse = 0.0

        weights_sum_nudges = []
        biases_sum_nudges = []

        for j in range(len(batch_id)):

            state = execute_model_full(model, images[batch_id[j]])
            expected_id = labels[batch_id[j]]
            expected = np.zeros(model.output_size)

            if model.output_mode == 0:
                expected[expected_id] = 1
            else:
                expected = class_to_binary(expected_id)

            batch_sse += np.sum((state[-1] - expected) ** 2)

            weights, biases = backpropagate(len(state) - 1, model, state, expected)

            for i in range(len(state) - 1):
                if len(weights_sum_nudges) <= i:
                    weights_sum_nudges.append(weights[i])
                else:
                    weights_sum_nudges[i] = weights_sum_nudges[i] + weights[i]
                if len(biases_sum_nudges) <= i:
                    biases_sum_nudges.append(biases[i])
                else:
                    biases_sum_nudges[i] = biases_sum_nudges[i] + biases[i]

        if len(weights_sum_nudges) > 0 and len(biases_sum_nudges) > 0:
            for i in range(len(model.layers) + 1):
                model.weights[i] += np.divide(weights_sum_nudges[i], len(batch_id)) * args.learning_rate
                model.biases[i] += np.divide(biases_sum_nudges[i], len(batch_id)) * args.learning_rate

        elapsed = time.time() - start

        if elapsed > cooldown:
            cooldown = np.ceil(elapsed)
            eta = elapsed / step * (args.steps - step)
            print(f"[{step}/{args.steps}] ({100 * step / args.steps:5.1f}%) | batch mean SSE: {batch_sse / len(batch_id):.6f} | elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    print(f"\nTraining finished in {time.time() - start:.1f}s")

    save_model(model, args.dest)
    print(f"Model saved to {args.dest}")

    return

def run(args):
    model = load_model(args.model)
    mode_name = "class" if model.output_mode == 0 else "binary"
    print(f"Loaded model {args.model} | layers: {[model.input_size] + model.layers + [model.output_size]} | output mode: {mode_name}")

    if args.folder:
        files = sorted(os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".dat"))
        if not files:
            print(f"No .dat files found in {args.input}")
            return
        print(f"Running {len(files)} file(s) from {args.input}")
    else:
        files = [args.input]

    results = [run_file(model, path, args.full) for path in files]

    if args.folder:
        valid = [r for r in results if r is not None]
        print(f"\n=== Summary: {sum(valid)}/{len(valid)} correct ({len(results) - len(valid)} skipped) ===")

    return

def extract(args):

    images = load_idx(f"{args.data}/images.idx")
    labels = load_idx(f"{args.data}/labels.idx")
    n = len(images)

    if args.index:
        indices = list(args.index)
    else:
        if args.count > n:
            raise ValueError(f"Cannot extract {args.count} samples from a dataset of {n}")
        if args.random:
            indices = np.random.choice(n, args.count, replace=False).tolist()
        else:
            indices = list(range(args.count))

    invalid = [i for i in indices if i < 0 or i >= n]
    if invalid:
        raise ValueError(f"Indices out of range (dataset size {n}): {invalid}")

    os.makedirs(args.out, exist_ok=True)
    print(f"Extracting {len(indices)} sample(s) from {args.data} to {args.out}")

    for idx in indices:
        label = int(labels[idx])
        filename = f"{args.name}.dat" if len(indices) == 1 else f"{args.name}_{idx}.dat"
        path = os.path.join(args.out, filename)

        with open(path, "wb") as f:
            f.write(struct.pack("B", label))
            f.write(images[idx].astype(np.uint8).tobytes())

        print(f"  sample {idx} | label {label} ('{emnist_class_to_char(label)}') -> {path}")

    print("Done.")
    return


####################################################################################
# ------------------------------- HELPER FUNCTIONS ------------------------------- #
####################################################################################

def run_file(model, path, full):

    with open(path, "rb") as f:
        raw = f.read()

    label = raw[0]
    pixels = np.frombuffer(raw[1:], dtype=np.uint8).astype(np.float32) / 255

    print(f"\n=== {path} ===")

    if len(pixels) != model.input_size:
        print(f"Skipped: expected {model.input_size} pixels, found {len(pixels)}")
        return None

    if full:
        state = execute_model_full(model, pixels)
        output = state[-1]
    else:
        output = execute_model(model, pixels)

    if model.output_mode == 0:
        predicted = int(np.argmax(output))
        interpretation = f"class {predicted} ('{emnist_class_to_char(predicted)}') with confidence {100 * output[predicted]:.2f}%"
    else:
        predicted = binary_to_class(output)   # raw outputs, no thresholding
        interpretation = f"outputs {np.round(output, 3).tolist()} -> class {predicted} ('{emnist_class_to_char(predicted)}')"

    match = predicted == label

    print(f"Output         : {output}")
    print(f"Interpretation : {interpretation}")
    print(f"Expected       : class {label} ('{emnist_class_to_char(label)}')")
    print(f"Result         : {'CORRECT' if match else 'WRONG'}")

    if full:
        print("\nNeurons:")
        for i, layer in enumerate(state):
            if i == 0:
                name = "Input"
            elif i == len(state) - 1:
                name = "Output"
            else:
                name = f"Hidden {i}"
            print(f"  {name} ({len(layer)} neurons): {format_neurons(layer)}")

    return match

def format_neurons(values):
    values = np.asarray(values).ravel()
    if len(values) > 24:
        shown = [f"{v:.4f}" for v in values[:12]] + ["..."] + [f"{v:.4f}" for v in values[-12:]]
    else:
        shown = [f"{v:.4f}" for v in values]
    return "[" + ", ".join(shown) + "]"

def backpropagate(layer, model, state, expected):

    if layer == 0:
        return ([], [])

    a_prev = state[layer-1]
    a = state[layer]
    w = model.weights[layer-1]
    b = model.biases[layer - 1]
    z = w @ a_prev + b

    biases_nudges = None

    ############## BIASES ##############

    if layer == (len(state) - 1) and model.output_mode == 0:
        s = np.sum(a*(a-expected))
        biases_nudges = 2*a*((a-expected)-s)
    elif layer == (len(state) - 1) or model.activation == 1:
        biases_nudges = dsigmoid(z)*2*(a - expected)
    elif model.activation == 0:
        biases_nudges = drelu(z)*2*(a - expected)
    elif model.activation == 2:
        biases_nudges = dtanh(z)*2*(a - expected)
    elif model.activation == 3:
        biases_nudges = dleaky_relu(z)*2*(a - expected)

    ############## WEIGHTS ##############

    weights_nudges = np.outer(biases_nudges, a_prev)

    ############## PREV_NEURONS ##############

    a_prev_nudges = w.T @ biases_nudges

    weights, biases = backpropagate(layer-1, model, state, a_prev - a_prev_nudges/2)
    weights.append(-weights_nudges)
    biases.append(-biases_nudges)
    return weights, biases

def load_model(model_path):

    with open(model_path, "rb") as f:
        layers_count = struct.unpack("B", f.read(1))[0]
        layers = [struct.unpack(">I", f.read(4))[0] for _ in range(layers_count)]
        input_size = struct.unpack(">I", f.read(4))[0]
        output_size = struct.unpack("B", f.read(1))[0]

        output_mode = struct.unpack("B", f.read(1))[0]

        if output_mode < 0 or output_mode > 1:
            raise ValueError(f"Invalid output mode in model file: {output_mode}")

        activation = struct.unpack("B", f.read(1))[0]

        leaky_relu_coef = 0.01

        if activation == 3:
            leaky_relu_coef = struct.unpack(">f", f.read(4))[0]

        full_layers = [input_size] + layers + [output_size]
        weights_matrices_sizes = [(full_layers[i+1], full_layers[i]) for i in range(len(full_layers) - 1)]

        weights = [np.frombuffer(f.read(np.prod(size) * 4), dtype=np.float32).reshape(size) for size in weights_matrices_sizes]

        biases = [np.frombuffer(f.read(size * 4), dtype=np.float32) for size in full_layers[1:]]

        model = Model(layers, input_size, output_size, output_mode, activation, leaky_relu_coef, weights, biases)

        return model

    return None

def save_model(model, model_path):
    with open(model_path, "wb") as f:
        # Write model parameters
        f.write(struct.pack("B", len(model.layers)))
        for layer in model.layers:
            f.write(struct.pack(">I", layer))
        f.write(struct.pack(">I", model.input_size))
        f.write(struct.pack("B", model.output_size))
        f.write(struct.pack("B", model.output_mode))
        f.write(struct.pack("B", model.activation))

        if model.activation == 3:
            f.write(struct.pack(">f", model.leaky_relu_coef))

        # Write weights and biases in float32 format
        for weight in model.weights:
            f.write(weight.astype(np.float32).tobytes())
        for bias in model.biases:
            f.write(bias.astype(np.float32).tobytes())

def generate_model_weights(model):
    weights = []
    biases = []

    full_layers = [model.input_size] + model.layers + [model.output_size]

    for i in range(len(full_layers) - 1):

        # Initialize weights with small random values
        weight_matrix = (np.random.randn(full_layers[i + 1], full_layers[i]) * np.sqrt(2.0 / full_layers[i])).astype(np.float32)
        bias_vector = np.zeros(full_layers[i + 1])

        weights.append(weight_matrix)
        biases.append(bias_vector)

    model.weights = weights
    model.biases = biases

def execute_model(model, input_data):
    data = np.array(input_data).flatten() 
    for i in range(len(model.layers)):
        data = model.weights[i] @ data + model.biases[i]
        if model.activation == 0:
            data = relu(data)
        elif model.activation == 1:
            data = sigmoid(data)
        elif model.activation == 2:
            data = tanh(data)
        elif model.activation == 3:
            data = leaky_relu(data, model.leaky_relu_coef)

    data = model.weights[len(model.layers)] @ data + model.biases[len(model.layers)]
    if model.output_mode == 0:
        data = softmax(data)
    else:
        data = sigmoid(data)
    return data

def execute_model_full(model, input_data):
    data = [np.array(input_data).flatten()]
    for i in range(len(model.layers)):
        data.append(model.weights[i] @ data[-1] + model.biases[i])
        if model.activation == 0:
            data[-1] = relu(data[-1])
        elif model.activation == 1:
            data[-1] = sigmoid(data[-1])
        elif model.activation == 2:
            data[-1] = tanh(data[-1])
        elif model.activation == 3:
            data[-1] = leaky_relu(data[-1], model.leaky_relu_coef)

    data.append(model.weights[len(model.layers)] @ data[-1] + model.biases[len(model.layers)])
    if model.output_mode == 0:
        data[-1] = softmax(data[-1])
    else:
        data[-1] = sigmoid(data[-1])
    return data


######### Activation functions ######### 

def relu(data):
    return np.maximum(0, data)

def leaky_relu(data, a):
    return np.where(data>0, data, data*a)

def tanh(data):
    return np.tanh(data)

def sigmoid(data):
    return 1/(1 + np.exp(-data))

def softmax(data):
    e = np.exp(data - np.max(data))
    s = sum(e)
    return np.array([e[i]/s for i in range(len(data))])

######### Activation functions derivatives ######### 

def drelu(data):
    return np.where(data>0, 1, 0)

def dleaky_relu(data, a):
    return np.where(data>0, 1, a)

def dtanh(data):
    return 1/(np.cosh(data)**2)

def dsigmoid(data):
    return sigmoid(data) * (1 - sigmoid(data))

def dsoftmax(data):
    s = softmax(data)
    return np.diag(s) - np.outer(s, s)

#####################################################

def load_idx(path):
    with open(path, "rb") as f:
        # Read IDX header
        magic = struct.unpack(">I", f.read(4))[0]

        dtype_code = (magic >> 8) & 0xFF
        ndim = magic & 0xFF

        dtype_map = {
            0x08: np.uint8,
            0x09: np.int8,
            0x0B: np.int16,
            0x0C: np.int32,
            0x0D: np.float32,
            0x0E: np.float64
        }

        dtype = dtype_map[dtype_code]

        # Read dimensions
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))

        # Read data
        data = np.frombuffer(f.read(), dtype=dtype)

        # IDX uses big-endian byte ordering
        dtype = np.dtype(dtype).newbyteorder(">")
        data = data.astype(dtype.newbyteorder("="))

        return data.reshape(shape)

def load_idx_normalized(path):
    with open(path, "rb") as f:
        # Read IDX header
        magic = struct.unpack(">I", f.read(4))[0]

        dtype_code = (magic >> 8) & 0xFF
        ndim = magic & 0xFF

        dtype_map = {
            0x08: np.uint8,
            0x09: np.int8,
            0x0B: np.int16,
            0x0C: np.int32,
            0x0D: np.float32,
            0x0E: np.float64
        }

        dtype = dtype_map[dtype_code]

        # Read dimensions
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))

        # Read data
        data = np.frombuffer(f.read(), dtype=dtype)

        # IDX uses big-endian byte ordering
        dtype = np.dtype(dtype).newbyteorder(">")
        data = data.astype(dtype.newbyteorder("="))

        data = np.divide(data.astype(np.float32), 255)

        return data.reshape(shape)

def emnist_class_to_char(label):
    if label < 10:
        return chr(label + ord('0'))
    elif label < 36:
        return chr(label - 10 + ord('A'))
    elif label < 62:
        return chr(label - 36 + ord('a'))
    elif label == 62:
        return '∅'
    else:
        raise ValueError(f"Invalid EMNIST class label: {label}")

def char_to_emnist_class(char):
    if '0' <= char <= '9':
        return ord(char) - ord('0')
    elif 'A' <= char <= 'Z':
        return ord(char) - ord('A') + 10
    elif 'a' <= char <= 'z':
        return ord(char) - ord('a') + 36
    elif char == '∅':
        return 62
    else:
        raise ValueError(f"Invalid character for EMNIST: {char}")

def class_to_binary(label): #convert a class id (0-62) to a binary vector with a humming distance of at least 3 between each class
    if label < 0 or label > 62:
        raise ValueError(f"Invalid class label: {label}")
    return ENCODING_MAP[label]

def binary_to_class(vector):
    """Return the class whose codeword is closest to `vector`.
    `vector` can hold raw sigmoid outputs in [0, 1]; no thresholding is needed."""
    vector = np.asarray(vector, dtype=np.float32).ravel()

    if len(vector) != 10:
        raise ValueError("Invalid binary vector length")

    # Squared Euclidean distance to every codeword at once
    distances = np.sum((np.array(ENCODING_MAP, dtype=np.float32) - vector) ** 2, axis=1)
    return int(np.argmin(distances))

class Model:
    def __init__(self, layers, input_size, output_size, output_mode, activation, leaky_relu_coef=0.01, weights=None, biases=None):
        self.layers = layers
        self.input_size = input_size
        self.output_size = output_size
        self.output_mode = output_mode
        self.activation = activation
        self.weights = weights
        self.biases = biases
        self.leaky_relu_coef = leaky_relu_coef

####################################################################################
####################################################################################
####################################################################################

if __name__ == "__main__":
    main()