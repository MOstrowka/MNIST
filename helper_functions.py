import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def process_image(image_path, img_size=32):
    """
    Funkcja do wczytywania i przetwarzania obrazu.
    
    Args:
    image_path (str): Ścieżka do obrazu.
    img_size (int): Rozmiar, do którego obraz ma być zmieniony.
    
    Returns:
    tf.Tensor: Przetworzony obraz.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = image / 255.0  # Normalizacja wartości pikseli
    return image

def prepare_dataset_simple(dataframe, path_column, label_column, img_size=32, batch_size=32):
    """
    Funkcja do przygotowania datasetu z ramki danych.
    
    Args:
    dataframe (pd.DataFrame): DataFrame zawierający ścieżki do obrazów i etykiety.
    path_column (str): Nazwa kolumny w DataFrame zawierającej ścieżki do obrazów.
    label_column (str): Nazwa kolumny w DataFrame zawierającej etykiety.
    img_size (int): Rozmiar, do którego mają być zmieniane obrazy.
    batch_size (int): Rozmiar batcha używany podczas tworzenia datasetu.
    
    Returns:
    tf.data.Dataset: Przygotowany dataset do trenowania modelu.
    """
    image_paths = dataframe[path_column].values
    labels = dataframe[label_column].values

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (process_image(x, img_size), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def display_random_images_with_predictions(test_dataset, y_true, y_pred_probs, total_images=9):
    """
    Wyświetla losowe obrazy testowe wraz z przewidywaniami i etykietami prawdziwymi
    na podstawie wcześniej wykonanych predykcji i datasetu TensorFlow.

    Args:
    test_dataset: Dataset TensorFlow zawierający obrazy testowe.
    y_true: Prawdziwe etykiety obrazów.
    y_pred_probs: Prawdopodobieństwa przewidziane przez model dla każdej klasy.
    total_images: Całkowita liczba obrazów do wyświetlenia.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    random_indices = np.random.choice(range(len(y_true)), size=total_images, replace=False)
    
    plt.figure(figsize=(12, 8))
    i = 1
    for index in random_indices:
        for img_index, (image, label) in enumerate(test_dataset.unbatch().as_numpy_iterator()):
            if img_index == index:
                plt.subplot(3, 3, i)
                plt.imshow(image)
                plt.title(f"True: {label}, Pred: {y_pred[index]}")
                plt.axis("off")
                i += 1
                break
    plt.tight_layout()
    plt.show()


def display_misclassified_images(test_dataset, y_true, y_pred, total_images=9):
    """
    Wyświetla określoną liczbę błędnie sklasyfikowanych obrazów na podstawie wcześniej wykonanych predykcji
    i datasetu TensorFlow.

    Args:
    test_dataset: Dataset TensorFlow zawierający obrazy testowe.
    y_true: Prawdziwe etykiety obrazów.
    y_pred: Etykiety przewidziane przez model.
    total_images: Całkowita liczba błędnie sklasyfikowanych obrazów do wyświetlenia.
    """
    misclassified_indices = np.where(y_pred != y_true)[0]
    selected_indices = np.random.choice(misclassified_indices, size=total_images, replace=False)
    
    plt.figure(figsize=(10, 10))
    i = 1
    for index in selected_indices:
        for img_index, (image, label) in enumerate(test_dataset.unbatch().as_numpy_iterator()):
            if img_index == index:
                plt.subplot(3, 3, i)
                plt.imshow(image)
                plt.title(f"True: {label}, Pred: {y_pred[index]}")
                plt.axis("off")
                i += 1
                break
    plt.tight_layout()
    plt.show()


def calculate_and_plot_roc_curve(y_true, y_pred_probs, n_classes):
    """
    Oblicza i rysuje krzywą ROC dla makro- i mikrośredniej.
    """
    # Binarizacja etykiet
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))
    
    # Obliczanie mikrośredniej ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_binarized.ravel(), y_pred_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # Obliczanie makrośredniej ROC
    roc_auc_macro = roc_auc_score(y_true_binarized, y_pred_probs, average='macro')
    fpr_macro, tpr_macro, _ = roc_curve(y_true_binarized.ravel(), y_pred_probs.ravel(), pos_label=1)
    
    # Rysowanie krzywej ROC dla mikrośredniej
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_micro, tpr_micro, label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc_micro))

    # Rysowanie krzywej ROC dla makrośredniej
    plt.plot(fpr_macro, tpr_macro, label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc_macro), linestyle='--')

    # Linia odniesienia
    plt.plot([0, 1], [0, 1], 'k--')

    # Ustawienia wykresu
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def calculate_and_plot_precision_recall_curve(y_true, y_pred_probs, n_classes):
    """
    Oblicza i rysuje krzywą Precision-Recall.
    """
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_pred_probs[:, i])
        average_precision[i] = average_precision_score(y_true_binarized[:, i], y_pred_probs[:, i])

    precision["macro"], recall["macro"], _ = precision_recall_curve(y_true_binarized.ravel(), y_pred_probs.ravel())
    average_precision["macro"] = average_precision_score(y_true_binarized, y_pred_probs, average="macro")

    plt.figure(figsize=(8, 6))
    plt.plot(recall['macro'], precision['macro'], label='Macro-average Precision-Recall curve (average precision = {0:0.2f})'.format(average_precision['macro']))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Macro-average Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


def plot_training_history_class(history):
    """
    Wizualizuje historię dokładności i straty podczas treningu i walidacji.

    Args:
    history: Zwrócony obiekt History z metody fit() modelu.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()