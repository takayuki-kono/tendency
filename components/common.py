import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

def get_preprocessing_function(model_name):
    preprocess_map = {
        'EfficientNetV2B0': efficientnet_preprocess,
        'EfficientNetV2S': efficientnet_preprocess,
        'ResNet50V2': resnet_preprocess,
        'Xception': xception_preprocess,
        'DenseNet121': densenet_preprocess
    }
    return preprocess_map.get(model_name, efficientnet_preprocess)

class BalancedSparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='balanced_accuracy', **kwargs):
        super(BalancedSparseCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.total_count = self.add_weight(name='tc', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)
        
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        if sample_weight is not None:
             sample_weight = tf.cast(sample_weight, tf.float32)
             sample_weight = tf.reshape(sample_weight, [-1])

        y_true_onehot = tf.one_hot(y_true, self.num_classes)
        
        if sample_weight is not None:
             sample_weight = tf.expand_dims(sample_weight, -1)
             y_true_onehot = y_true_onehot * sample_weight

        self.total_count.assign_add(tf.reduce_sum(y_true_onehot, axis=0))

        correct_mask = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        correct_onehot = y_true_onehot * tf.expand_dims(correct_mask, -1)
        self.true_positives.assign_add(tf.reduce_sum(correct_onehot, axis=0))

    def result(self):
        per_class_acc = tf.math.divide_no_nan(self.true_positives, self.total_count)
        return tf.reduce_mean(per_class_acc)

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.total_count.assign(tf.zeros(self.num_classes))

class MinClassAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='min_class_accuracy', **kwargs):
        super(MinClassAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.total_count = self.add_weight(name='tc', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle One-hot inputs (Mixup) - robust check
        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
            
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)
        
        # Flatten
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        if sample_weight is not None:
             sample_weight = tf.cast(sample_weight, tf.float32)
             sample_weight = tf.reshape(sample_weight, [-1])

        # Vectorized Update
        y_true_onehot = tf.one_hot(y_true, self.num_classes) # [Batch, NumClasses]
        
        if sample_weight is not None:
             # Ensure shape broadcasting [Batch, 1] * [Batch, NumClasses]
             sample_weight = tf.expand_dims(sample_weight, -1)
             y_true_onehot = y_true_onehot * sample_weight

        # Update Total Count (per class presence)
        self.total_count.assign_add(tf.reduce_sum(y_true_onehot, axis=0))

        # Update True Positives
        correct_mask = tf.cast(tf.equal(y_true, y_pred), tf.float32) # [Batch]
        # Mask the one-hot vectors to only keep correct ones
        correct_onehot = y_true_onehot * tf.expand_dims(correct_mask, -1)
        self.true_positives.assign_add(tf.reduce_sum(correct_onehot, axis=0))

    def result(self):
        # Calculate per-class accuracy
        per_class_acc = tf.math.divide_no_nan(self.true_positives, self.total_count)
        
        # Mask out classes with zero samples (total_count == 0)
        # We want min accuracy ONLY for classes that actually appeared in the validtion set
        mask = tf.cast(tf.not_equal(self.total_count, 0), tf.float32)
        
        # If no classes appeared, return 0.0
        if tf.reduce_sum(mask) == 0:
            return 0.0
            
        # Set accuracy of missing classes to 2.0 (or > 1.0) so they don't affect min
        # (Since accuracy is 0-1, anything > 1 is ignored by reduce_min)
        safe_per_class_acc = per_class_acc + (1.0 - mask) * 2.0
        
        return tf.reduce_min(safe_per_class_acc)

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.total_count.assign(tf.zeros(self.num_classes))
