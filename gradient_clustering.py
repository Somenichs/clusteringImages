import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import time
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class GradientImageClustering:
    """
    Класс для кластеризации изображений методом градиентного спуска
    с использованием soft K-means подхода
    """

    def __init__(self, n_clusters=3, learning_rate=0.01, max_iter=100,
                 tolerance=1e-4, sigma=1.0, decay_rate=0.001, batch_size=None,
                 init_method='kmeans++', random_state=42):
        """
        Параметры:
        - n_clusters: количество кластеров
        - learning_rate: начальная скорость обучения
        - max_iter: максимальное количество итераций
        - tolerance: порог сходимости
        - sigma: параметр размытости для soft assignment
        - decay_rate: скорость затухания learning rate
        - batch_size: размер батча (None для полного батча)
        - init_method: метод инициализации ('random' или 'kmeans++')
        - random_state: seed для воспроизводимости
        """
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.init_method = init_method
        self.random_state = random_state

        # Результаты обучения
        self.centroids = None
        self.labels = None
        self.loss_history = []
        self.lr_history = []
        self.converged = False
        self.n_iter = 0

        np.random.seed(random_state)

    def _kmeans_plus_plus_init(self, X):
        """
        Инициализация центроидов методом K-means++
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        # Выбираем первый центроид случайно
        centroids[0] = X[np.random.randint(n_samples)]

        # Выбираем остальные центроиды
        for i in range(1, self.n_clusters):
            # Вычисляем расстояния до ближайших центроидов
            min_distances = np.full(n_samples, np.inf)
            for j in range(i):
                distances = np.sum((X - centroids[j])**2, axis=1)
                min_distances = np.minimum(min_distances, distances)

            # Выбираем следующий центроид с вероятностью пропорциональной квадрату расстояния
            probabilities = min_distances / np.sum(min_distances)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.random()

            for j, prob in enumerate(cumulative_probs):
                if r <= prob:
                    centroids[i] = X[j]
                    break

        return centroids

    def _initialize_centroids(self, X):
        """
        Инициализация центроидов
        """
        if self.init_method == 'kmeans++':
            return self._kmeans_plus_plus_init(X)
        else:
            # Случайная инициализация
            n_samples, n_features = X.shape
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            return X[indices].copy()

    def _soft_assignment(self, X, centroids):
        """
        Вычисление мягких присваиваний (soft assignments)
        """
        n_samples = X.shape[0]
        assignments = np.zeros((n_samples, self.n_clusters))

        for i in range(n_samples):
            distances = np.sum((X[i] - centroids)**2, axis=1)
            # Численно стабильная версия softmax
            exp_neg_dist = np.exp(-distances / (2 * self.sigma**2))
            assignments[i] = exp_neg_dist / np.sum(exp_neg_dist)

        return assignments

    def _compute_loss(self, X, centroids, assignments):
        """
        Вычисление функции потерь
        """
        loss = 0.0
        n_samples = X.shape[0]

        for i in range(n_samples):
            for k in range(self.n_clusters):
                distance_sq = np.sum((X[i] - centroids[k])**2)
                loss += assignments[i, k] * distance_sq

        return loss / n_samples

    def _compute_gradient(self, X, centroids, assignments):
        """
        Вычисление градиента по центроидам
        """
        n_features = X.shape[1]
        gradients = np.zeros((self.n_clusters, n_features))

        for k in range(self.n_clusters):
            weighted_sum = np.zeros(n_features)
            weight_sum = 0.0

            for i in range(X.shape[0]):
                weight = assignments[i, k]
                weighted_sum += weight * (centroids[k] - X[i])
                weight_sum += weight

            if weight_sum > 1e-8:  # Избегаем деления на ноль
                gradients[k] = 2 * weighted_sum / weight_sum

        return gradients

    def _get_batch(self, X, batch_size):
        """
        Получение случайного батча данных
        """
        if batch_size is None or batch_size >= X.shape[0]:
            return X

        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        return X[indices]

    def fit(self, X):
        """
        Обучение модели
        """
        print(f"Запуск кластеризации с {self.n_clusters} кластерами...")
        print(f"Размер данных: {X.shape}")
        print(f"Метод инициализации: {self.init_method}")
        print(f"Режим батчей: {'mini-batch' if self.batch_size else 'full batch'}")

        start_time = time.time()

        # Инициализация центроидов
        self.centroids = self._initialize_centroids(X)
        prev_centroids = self.centroids.copy()

        self.loss_history = []
        self.lr_history = []

        for iteration in range(self.max_iter):
            # Получаем батч данных
            X_batch = self._get_batch(X, self.batch_size)

            # Вычисляем мягкие присваивания
            assignments = self._soft_assignment(X_batch, self.centroids)

            # Вычисляем функцию потерь на полных данных для мониторинга
            full_assignments = self._soft_assignment(X, self.centroids)
            loss = self._compute_loss(X, self.centroids, full_assignments)
            self.loss_history.append(loss)

            # Вычисляем градиент на батче
            gradients = self._compute_gradient(X_batch, self.centroids, assignments)

            # Обновляем центроиды
            self.centroids -= self.learning_rate * gradients

            # Обновляем learning rate с экспоненциальным затуханием
            self.learning_rate = self.initial_lr * np.exp(-self.decay_rate * iteration)
            self.lr_history.append(self.learning_rate)

            # Проверяем сходимость
            centroid_shift = np.mean(np.sqrt(np.sum((self.centroids - prev_centroids)**2, axis=1)))

            if iteration % 10 == 0:
                print(f"Итерация {iteration:3d}: Loss = {loss:.6f}, "
                      f"LR = {self.learning_rate:.6f}, Shift = {centroid_shift:.6f}")

            if centroid_shift < self.tolerance:
                print(f"\nСходимость достигнута на итерации {iteration}")
                self.converged = True
                break

            prev_centroids = self.centroids.copy()

        self.n_iter = iteration + 1

        # Финальные присваивания
        final_assignments = self._soft_assignment(X, self.centroids)
        self.labels = np.argmax(final_assignments, axis=1)

        end_time = time.time()
        print(f"\nОбучение завершено за {end_time - start_time:.2f} секунд")
        print(f"Количество итераций: {self.n_iter}")
        print(f"Финальная функция потерь: {self.loss_history[-1]:.6f}")

        return self

    def predict(self, X):
        """
        Предсказание кластеров для новых данных
        """
        if self.centroids is None:
            raise ValueError("Модель не обучена! Сначала вызовите fit()")

        assignments = self._soft_assignment(X, self.centroids)
        return np.argmax(assignments, axis=1)

    def get_confidence_map(self, X):
        """
        Вычисление карты уверенности
        """
        if self.centroids is None:
            raise ValueError("Модель не обучена!")

        assignments = self._soft_assignment(X, self.centroids)
        max_probs = np.max(assignments, axis=1)
        second_max_probs = np.partition(assignments, -2, axis=1)[:, -2]

        confidence = max_probs - second_max_probs
        return confidence


class ImageProcessor:
    """
    Класс для обработки изображений
    """

    @staticmethod
    def load_image(filepath):
        """
        Загрузка изображения
        """
        try:
            image = Image.open(filepath)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            raise ValueError(f"Ошибка загрузки изображения: {e}")

    @staticmethod
    def preprocess_image(image, normalize=True):
        """
        Предобработка изображения
        """
        # Преобразование в массив пикселей
        h, w, c = image.shape
        pixels = image.reshape(-1, c).astype(np.float32)

        if normalize:
            # Min-max нормализация
            pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min())

        return pixels, (h, w)

    @staticmethod
    def reconstruct_image(pixels, shape, labels, centroids):
        """
        Восстановление изображения из кластеров
        """
        h, w = shape
        clustered_pixels = centroids[labels]

        # Денормализация если нужно
        if clustered_pixels.max() <= 1.0:
            clustered_pixels = (clustered_pixels * 255).astype(np.uint8)

        return clustered_pixels.reshape(h, w, -1)


class MetricsCalculator:
    """
    Класс для вычисления метрик качества кластеризации
    """

    @staticmethod
    def silhouette_score_safe(X, labels):
        """
        Безопасное вычисление Silhouette Score
        """
        if len(np.unique(labels)) < 2:
            return 0.0
        try:
            return silhouette_score(X, labels)
        except:
            return 0.0

    @staticmethod
    def davies_bouldin_index(X, labels, centroids):
        """
        Вычисление Davies-Bouldin Index
        """
        n_clusters = len(centroids)
        if n_clusters < 2:
            return 0.0

        # Внутрикластерные расстояния
        cluster_distances = []
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                distances = np.sqrt(np.sum((cluster_points - centroids[i])**2, axis=1))
                cluster_distances.append(np.mean(distances))
            else:
                cluster_distances.append(0.0)

        # Davies-Bouldin Index
        db_index = 0.0
        for i in range(n_clusters):
            max_ratio = 0.0
            for j in range(n_clusters):
                if i != j:
                    inter_distance = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
                    if inter_distance > 0:
                        ratio = (cluster_distances[i] + cluster_distances[j]) / inter_distance
                        max_ratio = max(max_ratio, ratio)
            db_index += max_ratio

        return db_index / n_clusters

    @staticmethod
    def inertia(X, labels, centroids):
        """
        Вычисление инерции (внутрикластерной суммы квадратов)
        """
        total_inertia = 0.0
        for i, centroid in enumerate(centroids):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                distances_sq = np.sum((cluster_points - centroid)**2, axis=1)
                total_inertia += np.sum(distances_sq)
        return total_inertia


class Visualizer:
    """
    Класс для визуализации результатов
    """

    @staticmethod
    def plot_results(original_image, clustered_image, centroids, labels,
                    loss_history, confidence_map=None, shape=None):
        """
        Комплексная визуализация результатов
        """
        n_plots = 5 if confidence_map is not None else 4
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Результаты кластеризации изображения', fontsize=16)

        # Исходное изображение
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Исходное изображение')
        axes[0, 0].axis('off')

        # Результат кластеризации
        axes[0, 1].imshow(clustered_image)
        axes[0, 1].set_title('Кластеризованное изображение')
        axes[0, 1].axis('off')

        # Палитра цветов (центроиды)
        Visualizer._plot_color_palette(axes[0, 2], centroids)

        # Карта кластеров
        if shape:
            h, w = shape
            cluster_map = labels.reshape(h, w)
            im = axes[1, 0].imshow(cluster_map, cmap='tab10')
            axes[1, 0].set_title('Карта кластеров')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

        # График функции потерь
        axes[1, 1].plot(loss_history)
        axes[1, 1].set_title('Сходимость функции потерь')
        axes[1, 1].set_xlabel('Итерация')
        axes[1, 1].set_ylabel('Потери')
        axes[1, 1].grid(True)

        # Карта уверенности
        if confidence_map is not None and shape:
            h, w = shape
            conf_map = confidence_map.reshape(h, w)
            im = axes[1, 2].imshow(conf_map, cmap='viridis')
            axes[1, 2].set_title('Карта уверенности')
            axes[1, 2].axis('off')
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
        else:
            axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_color_palette(ax, centroids):
        """
        Отображение палитры цветов (центроидов)
        """
        n_clusters = len(centroids)

        # Нормализация цветов для отображения
        colors = centroids.copy()
        if colors.max() <= 1.0:
            colors = colors
        else:
            colors = colors / 255.0

        # Создание палитры
        palette = np.ones((100, n_clusters * 100, 3))
        for i, color in enumerate(colors):
            palette[:, i*100:(i+1)*100] = color

        ax.imshow(palette)
        ax.set_title('Палитра кластеров')
        ax.axis('off')

        # Добавляем подписи с RGB значениями
        for i, color in enumerate(centroids):
            x_pos = i * 100 + 50
            if colors.max() <= 1.0:
                rgb_text = f'RGB:\n({color[0]:.2f},\n{color[1]:.2f},\n{color[2]:.2f})'
            else:
                rgb_text = f'RGB:\n({int(color[0])},\n{int(color[1])},\n{int(color[2])})'
            ax.text(x_pos, 50, rgb_text, ha='center', va='center',
                   fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    @staticmethod
    def plot_comparison_with_kmeans(original_image, gradient_result, kmeans_result,
                                   gradient_centroids, kmeans_centroids):
        """
        Сравнение с классическим K-means
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Сравнение методов кластеризации', fontsize=16)

        # Исходное изображение
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Исходное изображение')
        axes[0, 0].axis('off')

        # Градиентный метод
        axes[0, 1].imshow(gradient_result)
        axes[0, 1].set_title('Градиентный спуск')
        axes[0, 1].axis('off')

        # K-means
        axes[0, 2].imshow(kmeans_result)
        axes[0, 2].set_title('Классический K-means')
        axes[0, 2].axis('off')

        # Палитры
        Visualizer._plot_color_palette(axes[1, 0], gradient_centroids)
        axes[1, 0].set_title('Палитра (Градиентный)')

        Visualizer._plot_color_palette(axes[1, 1], kmeans_centroids)
        axes[1, 1].set_title('Палитра (K-means)')

        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()


def elbow_method(X, max_clusters=10):
    """
    Метод локтя для определения оптимального количества кластеров
    """
    print("Выполнение метода локтя...")
    inertias = []
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        model = GradientImageClustering(n_clusters=k, max_iter=50, learning_rate=0.05)
        model.fit(X)

        # Вычисляем инерцию
        inertia = MetricsCalculator.inertia(X, model.labels, model.centroids)
        inertias.append(inertia)
        print(f"K={k}: Inertia={inertia:.2f}")

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Количество кластеров (K)')
    plt.ylabel('Инерция')
    plt.title('Метод локтя для определения оптимального K')
    plt.grid(True)
    plt.show()

    return K_range, inertias


def main():
    """
    Основная функция программы
    """
    parser = argparse.ArgumentParser(description='Кластеризация изображений методом градиентного спуска')
    parser.add_argument('image_path', help='Путь к изображению')
    parser.add_argument('--clusters', '-k', type=int, default=3, help='Количество кластеров (по умолчанию: 3)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.02, help='Скорость обучения (по умолчанию: 0.01)')
    parser.add_argument('--max_iter', '-i', type=int, default=1000, help='Максимум итераций (по умолчанию: 100)')
    parser.add_argument('--sigma', '-s', type=float, default=0.08, help='Параметр размытости (по умолчанию: 1.0)')
    parser.add_argument('--tolerance', '-tol', type=float, default=1e-4, help='Порог сходимости (по умолчанию: 1e-4)')
    parser.add_argument('--decay', '-d', type=float, default=0.001, help='Скорость затухания LR (по умолчанию: 0.001)')
    parser.add_argument('--batch', '-b', type=int, default=50000, help='Размер батча (по умолчанию: полный батч None)')
    parser.add_argument('--init', choices=['random', 'kmeans++'], default='kmeans++', help='Метод инициализации')
    parser.add_argument('--show', nargs='*', choices=['all', 'result', 'palette', 'loss', 'confidence', 'comparison'],
                       default=['all'], help='Что показывать в визуализации')
    parser.add_argument('--compare_kmeans', action='store_true', help='Сравнить с классическим K-means')
    parser.add_argument('--elbow', action='store_true', help='Выполнить метод локтя')
    parser.add_argument('--save', help='Путь для сохранения результата')

    args = parser.parse_args()

    try:
        # Загрузка и предобработка изображения
        print("Загрузка изображения...")
        original_image = ImageProcessor.load_image(args.image_path)
        pixels, shape = ImageProcessor.preprocess_image(original_image, normalize=True)

        print(f"Размер изображения: {shape[1]}x{shape[0]} пикселей")
        print(f"Общее количество пикселей: {len(pixels)}")

        # Метод локтя, если запрошен
        if args.elbow:
            elbow_method(pixels)
            return

        # Создание и обучение модели
        model = GradientImageClustering(
            n_clusters=args.clusters,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
            sigma=args.sigma,
            decay_rate=args.decay,
            batch_size=args.batch,
            init_method=args.init
        )

        model.fit(pixels)

        # Восстановление изображения
        clustered_image = ImageProcessor.reconstruct_image(
            pixels, shape, model.labels, model.centroids
        )

        # Вычисление метрик
        print("\n=== МЕТРИКИ КАЧЕСТВА ===")
        silhouette = MetricsCalculator.silhouette_score_safe(pixels, model.labels)
        db_index = MetricsCalculator.davies_bouldin_index(pixels, model.labels, model.centroids)
        inertia = MetricsCalculator.inertia(pixels, model.labels, model.centroids)

        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {db_index:.4f}")
        print(f"Инерция: {inertia:.2f}")

        # Карта уверенности
        confidence_map = model.get_confidence_map(pixels)
        print(f"Средняя уверенность: {np.mean(confidence_map):.4f}")

        # Сравнение с K-means, если запрошено
        if args.compare_kmeans:
            print("\nСравнение с классическим K-means...")
            kmeans = KMeans(n_clusters=args.clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(pixels)
            kmeans_image = ImageProcessor.reconstruct_image(
                pixels, shape, kmeans_labels, kmeans.cluster_centers_
            )

            # Метрики для K-means
            kmeans_silhouette = MetricsCalculator.silhouette_score_safe(pixels, kmeans_labels)
            kmeans_db = MetricsCalculator.davies_bouldin_index(pixels, kmeans_labels, kmeans.cluster_centers_)
            kmeans_inertia = MetricsCalculator.inertia(pixels, kmeans_labels, kmeans.cluster_centers_)

            print(f"\nK-means метрики:")
            print(f"Silhouette Score: {kmeans_silhouette:.4f}")
            print(f"Davies-Bouldin Index: {kmeans_db:.4f}")
            print(f"Инерция: {kmeans_inertia:.2f}")

            # Визуализация сравнения
            Visualizer.plot_comparison_with_kmeans(
                original_image, clustered_image, kmeans_image,
                model.centroids * 255, kmeans.cluster_centers_ * 255
            )

        # Визуализация результатов
        if 'all' in args.show or not args.show:
            show_confidence = True
        else:
            show_confidence = 'confidence' in args.show

        Visualizer.plot_results(
            original_image, clustered_image,
            model.centroids * 255, model.labels,
            model.loss_history,
            confidence_map if show_confidence else None,
            shape
        )

        # Сохранение результата
        if args.save:
            result_image = Image.fromarray(clustered_image.astype(np.uint8))
            result_image.save(args.save)
            print(f"\nРезультат сохранен: {args.save}")

        print("\nКластеризация завершена успешно!")

    except Exception as e:
        print(f"Ошибка: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
