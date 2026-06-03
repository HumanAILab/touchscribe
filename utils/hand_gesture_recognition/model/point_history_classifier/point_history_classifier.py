import numpy as np
import tensorflow as tf
import os 

class PointHistoryClassifier(object):
    def __init__(
        self,
        model_path,
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
    ):
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script's directory
        MODEL_DIR = os.path.join(BASE_DIR, model_path)

        self.interpreter = tf.lite.Interpreter(model_path=MODEL_DIR,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([point_history], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index


# import numpy as np
# import os
# import tensorflow as tf


# def compute_velocity(point_history):
#     """Compute velocity for a 1D flattened point history (x1, y1, x2, y2, ...)."""
    
#     point_history = np.array(point_history, dtype=np.float32)

#     # Ensure length is even (each point has an (x, y) pair)
#     if point_history.shape[0] % 2 != 0:
#         raise ValueError(f"Unexpected point_history length: {point_history.shape[0]}. Expected even length for (x, y) pairs.")

#     # Compute differences for x and y separately
#     x_diff = np.diff(point_history[0::2])  # Take every other value starting at index 0 (x values)
#     y_diff = np.diff(point_history[1::2])  # Take every other value starting at index 1 (y values)

#     # Compute Euclidean distance (velocity)
#     velocity = np.sqrt(x_diff**2 + y_diff**2)  

#     return velocity

# class PointHistoryClassifier(object):
#     def __init__(self, model_path, score_th=0.5, invalid_value=-1, num_threads=1, velocity_th=0.05):
#         self.velocity_th = velocity_th  # Threshold for movement filtering

#         BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#         MODEL_DIR = os.path.join(BASE_DIR, model_path)

#         self.interpreter = tf.lite.Interpreter(model_path=MODEL_DIR, num_threads=num_threads)
#         self.finger = model_path.split('_')[0]
#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()

#         self.score_th = score_th
#         self.invalid_value = invalid_value

#     def __call__(self, point_history):
#         """Classify gesture only if movement is below a threshold."""
#         point_history = np.array(point_history, dtype=np.float32)

        

#         velocity = compute_velocity(point_history)
#         # print(f"{self.finger} velocity: {np.mean(velocity)}\n"*7)  # Debugging
#         if np.mean(velocity) > self.velocity_th:
#             return self.invalid_value

#         # Run inference
#         input_details_tensor_index = self.input_details[0]['index']
#         self.interpreter.set_tensor(
#             input_details_tensor_index,
#             np.array([point_history], dtype=np.float32)
#         )
#         self.interpreter.invoke()

#         output_details_tensor_index = self.output_details[0]['index']
#         result = self.interpreter.get_tensor(output_details_tensor_index)

#         result_index = np.argmax(np.squeeze(result))

#         # Apply confidence threshold
#         if np.squeeze(result)[result_index] < self.score_th:
#             result_index = self.invalid_value

#         return result_index
