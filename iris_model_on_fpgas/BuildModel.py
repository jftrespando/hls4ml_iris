from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_schedule, pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical
import hls4ml
import numpy as np
import os
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class BuildModel:
    """
    Initialize the BuildModel class.

    Args:
        seed (int): The seed for random number generation.
        board_ip (str): The IP address of the target board.
        board_username (str): The username for the target board.

    Returns:
        None
    """
    def __init__(self, seed=0, board_ip='192.168.68.126', board_username='xilinx'):
        self.seed = seed
        self.board_ip = board_ip
        self.board_username = board_username

        os.makedirs('package', exist_ok=True)

    def create_dataset(self):
        """
        Create a dataset for training the model.

        This function loads the Iris dataset, preprocesses it, and splits it into training and testing sets.
        The dataset is saved for later use in the deployment process.

        Returns:
            None
        """
        # Set the seed for random number generation
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Load the Iris dataset
        data = load_iris()
        X, y = data['data'], data['target']

        # Convert the target labels to categorical values and one-hot encode them
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = to_categorical(y, 3)  # Iris dataset has 3 classes

        # Split the dataset into training and testing sets
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features in the training and testing sets
        scaler = StandardScaler()
        self.X_train_val = scaler.fit_transform(self.X_train_val)
        self.X_test = scaler.transform(self.X_test)

        # Save the testing and classes set for later use
        np.save('package/classes.npy', le.classes_)
        np.save('package/X_test.npy', self.X_test)
        np.save('package/y_test.npy', self.y_test)

    def train_model(self):
        """
        Build, train, and save a pruned and quantized neural network model using QKeras.

        This function constructs a Sequential model with QKeras layers. The model is pruned to reduce the number of parameters
        and quantized to use lower precision for the weights and activations. The training process includes setting up
        callbacks for updating pruning steps. After training, the pruning wrappers are stripped, and the final model is saved.

        Parameters:
        None

        Returns:
        None
        """
        # Initialize a Sequential model
        self.model = Sequential()

        # Add the first QDense layer with quantization and pruning
        self.model.add(
            QDense(
                64,  # Number of neurons in the layer
                input_shape=(4,),  # Input shape for the Iris dataset (4 features)
                name='fc1',
                kernel_quantizer=quantized_bits(6, 0, alpha=1),  # Quantize weights to 6 bits
                bias_quantizer=quantized_bits(6, 0, alpha=1),  # Quantize biases to 6 bits
                kernel_initializer='lecun_uniform',
                kernel_regularizer=l1(0.0001),
            )
        )

        # Add a quantized ReLU activation layer
        self.model.add(QActivation(activation=quantized_relu(6), name='relu1'))

        # Add the second QDense layer with quantization and pruning
        self.model.add(
            QDense(
                32,  # Number of neurons in the layer
                name='fc2',
                kernel_quantizer=quantized_bits(6, 0, alpha=1),  # Quantize weights to 6 bits
                bias_quantizer=quantized_bits(6, 0, alpha=1),  # Quantize biases to 6 bits
                kernel_initializer='lecun_uniform',
                kernel_regularizer=l1(0.0001),
            )
        )

        # Add a quantized ReLU activation layer
        self.model.add(QActivation(activation=quantized_relu(6), name='relu2'))

        # Add the third QDense layer with quantization and pruning
        self.model.add(
            QDense(
                32,  # Number of neurons in the layer
                name='fc3',
                kernel_quantizer=quantized_bits(6, 0, alpha=1),  # Quantize weights to 6 bits
                bias_quantizer=quantized_bits(6, 0, alpha=1),  # Quantize biases to 6 bits
                kernel_initializer='lecun_uniform',
                kernel_regularizer=l1(0.0001),
            )
        )

        # Add a quantized ReLU activation layer
        self.model.add(QActivation(activation=quantized_relu(6), name='relu3'))

        # Add the output QDense layer with quantization and pruning
        self.model.add(
            QDense(
                3,  # Number of neurons in the output layer (matches the number of classes in the Iris dataset)
                name='output',
                kernel_quantizer=quantized_bits(6, 0, alpha=1),  # Quantize weights to 6 bits
                bias_quantizer=quantized_bits(6, 0, alpha=1),  # Quantize biases to 6 bits
                kernel_initializer='lecun_uniform',
                kernel_regularizer=l1(0.0001),
            )
        )

        # Add a softmax activation layer for classification
        self.model.add(Activation(activation='softmax', name='softmax'))

        # Set up pruning parameters to prune 75% of weights, starting after 2000 steps and updating every 100 steps
        pruning_params = {"pruning_schedule": pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)}
        self.model = prune.prune_low_magnitude(self.model, **pruning_params)

        # Compile the model with Adam optimizer and categorical crossentropy loss
        adam = Adam(lr=0.0001)
        self.model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])

        # Set up the pruning callback to update pruning steps during training
        callbacks = [
            pruning_callbacks.UpdatePruningStep(),
        ]

        # Train the model
        self.model.fit(
            self.X_train_val,
            self.y_train_val,
            batch_size=32,  # Batch size for training
            epochs=30,  # Number of training epochs
            validation_split=0.25,  # Fraction of training data to be used as validation data
            shuffle=True,  # Shuffle training data before each epoch
            callbacks=callbacks,  # Include pruning callback
        )

        # Strip the pruning wrappers from the model
        self.model = strip_pruning(self.model)

    def build_bitstream(self):
        """
        Builds the HLS bitstream for the trained model.

        This function converts the trained Keras model to an HLS model using hls4ml, compiles it, and generates the bitstream for FPGA deployment.
        It also validates the HLS model against the original model and prints the accuracy of both models.

        """
        # Create an HLS config from the Keras model, with the layer names granularity
        config = hls4ml.utils.config_from_keras_model(self.model, granularity='name')

        # Set precision for the softmax layer
        config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
        config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'

        # Set the ReuseFactor for the fully connected layers to 64
        for layer in ['fc1', 'fc2', 'fc3', 'output']:
            config['LayerName'][layer]['ReuseFactor'] = 64

        # Convert the Keras model to an HLS model
        hls_model = hls4ml.converters.convert_from_keras_model(
            self.model, hls_config=config, output_dir='hls4ml_prj_pynq', backend='VivadoAccelerator', board='pynq-z2'
        )

        # Compile the HLS model
        hls_model.compile()

        # Predict using the HLS model
        y_hls = hls_model.predict(np.ascontiguousarray(self.X_test))
        np.save('package/y_hls.npy', y_hls)

        # Validate the HLS model against the original model
        y_pred = self.model.predict(self.X_test)
        accuracy_original = accuracy_score(np.argmax(self.y_test, axis=1), np.argmax(y_pred, axis=1))
        accuracy_hls = accuracy_score(np.argmax(self.y_test, axis=1), np.argmax(y_hls, axis=1))

        print(f"Accuracy of the original pruned and quantized model: {accuracy_original * 100:.2f}%")
        print(f"Accuracy of the HLS model: {accuracy_hls * 100:.2f}%")

        # Build the HLS model
        hls_model.build(csim=False, export=True, bitfile=True)

    def prepare_files(self):
        """
        Prepares the files for deployment on the PYNQ board.

        Copies the necessary files to the 'package' directory.
        """
        os.system('cp hls4ml_prj_pynq/myproject_vivado_accelerator/project_1.runs/impl_1/design_1_wrapper.bit package/hls4ml_nn.bit')
        os.system('cp hls4ml_prj_pynq/myproject_vivado_accelerator/project_1.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh package/hls4ml_nn.hwh')
        os.system('cp hls4ml_prj_pynq/axi_stream_driver.py package/axi_stream_driver.py')
        os.system('cp utilities/on_target.py package/on_target.py')

    def copy_files(self):
        """
        Copies the necessary files to the PYNQ board.

        This function uses SCP to transfer the contents of the 'package' directory to the target board.
        Make sure to set up authentication first using `ssh-copy-id`.
        """
        os.system(f'scp -r package/* {self.board_username}@{self.board_ip}:/home/{self.board_username}/jupyter_notebooks/iris_model_on_fpgas')

def run():
    """
    Run the build process.

    This function creates a dataset, trains a model, builds a bitstream, prepares files, and copies files to the PYNQ board.
    """
    build = BuildModel()
    build.create_dataset()
    build.train_model()
    build.build_bitstream()
    build.prepare_files()
    build.copy_files()

if __name__ == '__main__':
    run()
