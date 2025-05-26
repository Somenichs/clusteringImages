import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
import sys
from functools import partial

# Опциональные импорты
try:
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-learn не найден - некоторые метрики будут недоступны")
    SKLEARN_AVAILABLE = False

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("✅ Numba доступна - ускорение включено")
except ImportError:
    print("⚠️  Numba не найдена - работа без JIT")
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

warnings.filterwarnings('ignore')

# Быстрые вычисления с прогрессом
class ProgressBar:
    """Простой прогресс-бар"""
    def __init__(self, total, description="Прогресс"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, step=1):
        self.current += step
        if self.current % max(1, self.total // 20) == 0 or self.current == self.total:
            progress = self.current / self.total
            elapsed = time.time() - self.start_time
            eta = elapsed / progress - elapsed if progress > 0 else 0

            bar_length = 30
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)

            print(f"\r{self.description}: [{bar}] {progress*100:.1f}% "
                  f"(ETA: {eta:.1f}s)", end="", flush=True)

            if self.current == self.total:
                print()  # Новая строка в конце

# Оптимизированные функции
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def fast_distances(pixels, centroids):
        n_pixels, n_features = pixels.shape
        n_clusters = centroids.shape[0]
        distances = np.zeros((n_pixels, n_clusters))

        for i in prange(n_pixels):
            for k in range(n_clusters):
                dist = 0.0
                for j in range(n_features):
                    diff = pixels[i, j] - centroids[k, j]
                    dist += diff * diff
                distances[i, k] = dist
        return distances

    @jit(nopython=True, parallel=True)
    def fast_assignments(distances, sigma):
        n_pixels, n_clusters = distances.shape
        assignments = np.zeros((n_pixels, n_clusters))

        for i in prange(n_pixels):
            min_dist = np.min(distances[i])
            exp_sum = 0.0
            exp_values = np.zeros(n_clusters)

            for k in range(n_clusters):
                exp_values[k] = np.exp(-(distances[i, k] - min_dist) / (2 * sigma * sigma))
                exp_sum += exp_values[k]

            for k in range(n_clusters):
                assignments[i, k] = exp_values[k] / exp_sum

        return assignments

    @jit(nopython=True)
    def fast_silhouette_sample(distances_to_centroids, labels, n_clusters):
        """Быстрая аппроксимация silhouette для выборки"""
        n_samples = len(labels)
        silhouette_values = np.zeros(n_samples)

        for i in range(n_samples):
            # Внутрикластерное расстояние (до своего центроида)
            own_cluster = labels[i]
            a_i = distances_to_centroids[i, own_cluster]

            # Ближайшее расстояние до другого кластера
            b_i = np.inf
            for k in range(n_clusters):
                if k != own_cluster:
                    if distances_to_centroids[i, k] < b_i:
                        b_i = distances_to_centroids[i, k]

            # Silhouette коэффициент
            if max(a_i, b_i) > 0:
                silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
            else:
                silhouette_values[i] = 0.0

        return silhouette_values

else:
    def fast_distances(pixels, centroids):
        return np.sum((pixels[:, np.newaxis] - centroids)**2, axis=2)

    def fast_assignments(distances, sigma):
        exp_values = np.exp(-distances / (2 * sigma**2))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def fast_silhouette_sample(distances_to_centroids, labels, n_clusters):
        n_samples = len(labels)
        silhouette_values = np.zeros(n_samples)

        for i in range(n_samples):
            own_cluster = labels[i]
            a_i = distances_to_centroids[i, own_cluster]

            other_distances = distances_to_centroids[i, [k for k in range(n_clusters) if k != own_cluster]]
            b_i = np.min(other_distances) if len(other_distances) > 0 else a_i

            if max(a_i, b_i) > 0:
                silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)

        return silhouette_values


class FastMultithreadClustering:
    """Быстрая многопоточная кластеризация с оптимизацией"""

    def __init__(self, n_clusters=3, learning_rate=0.01, max_iter=100,
                 tolerance=1e-4, sigma=1.0, n_threads=None, random_state=42):

        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.sigma = sigma
        self.random_state = random_state

        if n_threads is None:
            self.n_threads = min(mp.cpu_count(), 6)
        else:
            self.n_threads = min(n_threads, mp.cpu_count())

        self.centroids = None
        self.labels = None
        self.loss_history = []
        self.converged = False
        self.n_iter = 0

        np.random.seed(random_state)
        print(f"🚀 Быстрая кластеризация: {self.n_threads} потоков, {n_clusters} кластеров")

    def _kmeans_plus_plus_init(self, X):
        """Быстрая K-means++ инициализация"""
        print("🔄 Инициализация центроидов...")
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        centroids[0] = X[np.random.randint(n_samples)]

        for i in range(1, self.n_clusters):
            distances = np.array([
                min([np.sum((x - c)**2) for c in centroids[:i]])
                for x in X
            ])

            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.random()

            for j, prob in enumerate(cumulative_probs):
                if r <= prob:
                    centroids[i] = X[j]
                    break

        print("✅ Центроиды инициализированы")
        return centroids

    def _compute_batch_parallel(self, X_batch):
        """Быстрая обработка батча"""
        distances = fast_distances(X_batch, self.centroids)
        assignments = fast_assignments(distances, self.sigma)
        loss = np.sum(assignments * distances) / len(X_batch)
        return assignments, loss, distances

    def _compute_assignments_and_loss_fast(self, X):
        """Ультра-быстрое вычисление с прогрессом"""
        if len(X) < 2000:  # Маленькие данные - обрабатываем сразу
            return self._compute_batch_parallel(X)

        # Большие данные - батчами с прогрессом
        batch_size = max(500, len(X) // self.n_threads)
        batches = [X[i:i+batch_size] for i in range(0, len(X), batch_size)]

        progress = ProgressBar(len(batches), "🔄 Обработка батчей")

        def process_with_progress(batch_idx_and_data):
            batch_idx, batch_data = batch_idx_and_data
            result = self._compute_batch_parallel(batch_data)
            progress.update()
            return result

        # Параллельная обработка
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            batch_data = [(i, batch) for i, batch in enumerate(batches)]
            results = list(executor.map(process_with_progress, batch_data))

        # Объединяем результаты
        all_assignments = np.vstack([r[0] for r in results])
        avg_loss = np.mean([r[1] for r in results])
        all_distances = np.vstack([r[2] for r in results])

        return all_assignments, avg_loss, all_distances

    def fit(self, X):
        """Быстрое обучение"""
        print(f"\n🚀 Начало обучения на {len(X)} пикселях...")
        start_time = time.time()

        self.centroids = self._kmeans_plus_plus_init(X)
        prev_centroids = self.centroids.copy()
        self.loss_history = []

        for iteration in range(self.max_iter):
            # Быстрое вычисление присваиваний
            assignments, loss, _ = self._compute_assignments_and_loss_fast(X)
            self.loss_history.append(loss)

            # Градиенты
            gradients = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                weights = assignments[:, k:k+1]
                if np.sum(weights) > 1e-8:
                    weighted_diff = weights * (self.centroids[k] - X)
                    gradients[k] = 2 * np.mean(weighted_diff, axis=0)

            # Обновление
            self.centroids -= self.learning_rate * gradients
            self.learning_rate = self.initial_lr * np.exp(-0.001 * iteration)

            # Сходимость
            centroid_shift = np.mean(np.sqrt(np.sum((self.centroids - prev_centroids)**2, axis=1)))

            if iteration % 20 == 0:
                print(f"   Итерация {iteration:3d}: Loss={loss:.6f}, Shift={centroid_shift:.6f}")

            if centroid_shift < self.tolerance:
                print(f"✅ Сходимость на итерации {iteration}")
                self.converged = True
                break

            prev_centroids = self.centroids.copy()

        self.n_iter = iteration + 1

        # БЫСТРЫЕ финальные присваивания
        print("🔄 Финальные присваивания...")
        final_assignments, _, self.final_distances = self._compute_assignments_and_loss_fast(X)
        self.labels = np.argmax(final_assignments, axis=1)

        total_time = time.time() - start_time
        print(f"🎉 Обучение завершено за {total_time:.2f}с")
        print(f"   Итераций: {self.n_iter}")
        print(f"   Скорость: {len(X) * self.n_iter / total_time:.0f} пикс/сек")

        return self

    def get_fast_metrics(self, X, sample_size=5000):
        """Быстрое вычисление метрик на выборке"""
        print(f"📊 Быстрое вычисление метрик (выборка: {min(sample_size, len(X))})...")

        # Используем выборку для больших данных
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            sample_X = X[indices]
            sample_labels = self.labels[indices]
            sample_distances = self.final_distances[indices]
        else:
            sample_X = X
            sample_labels = self.labels
            sample_distances = self.final_distances

        metrics = {}

        # Быстрая аппроксимация Silhouette Score
        if SKLEARN_AVAILABLE and len(np.unique(sample_labels)) > 1:
            try:
                print("🔄 Вычисление Silhouette Score...")
                if NUMBA_AVAILABLE:
                    # Используем быструю аппроксимацию
                    silhouette_values = fast_silhouette_sample(
                        sample_distances, sample_labels, self.n_clusters
                    )
                    metrics['silhouette_score'] = np.mean(silhouette_values)
                else:
                    # Fallback на sklearn с выборкой
                    if len(sample_X) > 1000:  # Еще больше уменьшаем выборку
                        sub_indices = np.random.choice(len(sample_X), 1000, replace=False)
                        metrics['silhouette_score'] = silhouette_score(
                            sample_X[sub_indices], sample_labels[sub_indices]
                        )
                    else:
                        metrics['silhouette_score'] = silhouette_score(sample_X, sample_labels)
            except Exception as e:
                print(f"⚠️  Ошибка Silhouette Score: {e}")
                metrics['silhouette_score'] = 0.0
        else:
            metrics['silhouette_score'] = 0.0

        # Быстрая карта уверенности (только для части данных)
        print("🔄 Карта уверенности...")
        confidence_sample_size = min(1000, len(X))
        if len(X) > confidence_sample_size:
            conf_indices = np.random.choice(len(X), confidence_sample_size, replace=False)
            conf_distances = self.final_distances[conf_indices]
        else:
            conf_distances = self.final_distances

        # Быстрое вычисление уверенности
        conf_assignments = fast_assignments(conf_distances, self.sigma)
        sorted_probs = np.sort(conf_assignments, axis=1)
        confidence_sample = sorted_probs[:, -1] - sorted_probs[:, -2]
        metrics['avg_confidence'] = np.mean(confidence_sample)

        print("✅ Метрики вычислены быстро!")
        return metrics

    def create_fast_visualization(self, original_image, pixels, shape, save_only=False):
        """Быстрая визуализация без зависаний"""
        print("📊 Быстрая визуализация...")

        try:
            # Быстрое восстановление изображения
            clustered_pixels = self.centroids[self.labels]
            clustered_image = (clustered_pixels * 255).astype(np.uint8).reshape(original_image.shape)

            if save_only:
                # Только сохранение без показа
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle('Быстрая кластеризация', fontsize=14)

                axes[0].imshow(original_image)
                axes[0].set_title('Исходное')
                axes[0].axis('off')

                axes[1].imshow(clustered_image)
                axes[1].set_title('Результат')
                axes[1].axis('off')

                plt.tight_layout()
                plt.savefig('clustering_result.png', dpi=150, bbox_inches='tight')
                plt.close()  # Важно закрыть!

                print("✅ Результат сохранен в clustering_result.png")
            else:
                # Полная визуализация
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle('🚀 Быстрая многопоточная кластеризация', fontsize=16)

                # Исходное изображение
                axes[0, 0].imshow(original_image)
                axes[0, 0].set_title('Исходное изображение')
                axes[0, 0].axis('off')

                # Результат
                axes[0, 1].imshow(clustered_image)
                axes[0, 1].set_title('Результат кластеризации')
                axes[0, 1].axis('off')

                # Палитра
                palette = np.ones((50, self.n_clusters * 50, 3))
                for i, color in enumerate(self.centroids):
                    palette[:, i*50:(i+1)*50] = color

                axes[0, 2].imshow(palette)
                axes[0, 2].set_title('Палитра')
                axes[0, 2].axis('off')

                # Карта кластеров (уменьшенная для скорости)
                cluster_map = self.labels.reshape(shape)
                if max(shape) > 500:  # Уменьшаем большие изображения
                    from PIL import Image as PILImage
                    cluster_img = PILImage.fromarray((cluster_map * 25).astype(np.uint8))
                    cluster_img = cluster_img.resize((min(500, shape[1]), min(500, shape[0])))
                    cluster_map = np.array(cluster_img)

                im1 = axes[1, 0].imshow(cluster_map, cmap='tab10')
                axes[1, 0].set_title('Карта кластеров')
                axes[1, 0].axis('off')

                # График сходимости
                axes[1, 1].plot(self.loss_history, 'b-', linewidth=2)
                axes[1, 1].set_title('Сходимость')
                axes[1, 1].set_xlabel('Итерация')
                axes[1, 1].set_ylabel('Потери')
                axes[1, 1].grid(True)

                # Статистика
                axes[1, 2].text(0.1, 0.8, f'Кластеров: {self.n_clusters}', transform=axes[1, 2].transAxes, fontsize=12)
                axes[1, 2].text(0.1, 0.7, f'Итераций: {self.n_iter}', transform=axes[1, 2].transAxes, fontsize=12)
                axes[1, 2].text(0.1, 0.6, f'Пикселей: {len(pixels)}', transform=axes[1, 2].transAxes, fontsize=12)
                axes[1, 2].text(0.1, 0.5, f'Потоков: {self.n_threads}', transform=axes[1, 2].transAxes, fontsize=12)
                axes[1, 2].text(0.1, 0.4, f'Сходимость: {"Да" if self.converged else "Нет"}', transform=axes[1, 2].transAxes, fontsize=12)
                axes[1, 2].set_title('Статистика')
                axes[1, 2].axis('off')

                plt.tight_layout()

                # Показываем с тайм-аутом
                try:
                    plt.show(block=False)
                    plt.pause(0.1)  # Небольшая пауза для отрисовки
                    print("✅ Визуализация создана")

                    # Сохраняем копию
                    plt.savefig('clustering_result.png', dpi=150, bbox_inches='tight')
                    print("💾 Результат сохранен в clustering_result.png")

                except Exception as e:
                    print(f"⚠️  Предупреждение при показе: {e}")
                    plt.savefig('clustering_result.png', dpi=150, bbox_inches='tight')
                    print("💾 Результат сохранен в clustering_result.png")

        except Exception as e:
            print(f"❌ Ошибка визуализации: {e}")
            print("💾 Попытка базового сохранения...")

            # Минимальное сохранение
            clustered_pixels = self.centroids[self.labels]
            clustered_image = (clustered_pixels * 255).astype(np.uint8).reshape(original_image.shape)
            Image.fromarray(clustered_image).save('clustering_result_basic.png')
            print("✅ Базовый результат сохранен")


def create_test_image(filename="test_fast.png"):
    """Быстрое создание тестового изображения"""
    print(f"🎨 Создание тестового изображения: {filename}")

    img = Image.new('RGB', (300, 300), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # Простые фигуры
    draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0))
    draw.ellipse([150, 150, 250, 250], fill=(0, 255, 0))
    draw.rectangle([20, 200, 80, 280], fill=(0, 0, 255))
    draw.rectangle([220, 20, 280, 80], fill=(255, 255, 0))

    img.save(filename)
    print(f"✅ Создано: {filename}")
    return filename


def main():
    """Быстрая основная функция"""
    parser = argparse.ArgumentParser(description='БЫСТРАЯ многопоточная кластеризация')
    parser.add_argument('image_path', nargs='?', help='Путь к изображению')
    parser.add_argument('--clusters', '-k', type=int, default=3, help='Количество кластеров')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Скорость обучения')
    parser.add_argument('--max_iter', '-i', type=int, default=100, help='Максимум итераций')
    parser.add_argument('--threads', '-t', type=int, default=11, help='Количество потоков')
    parser.add_argument('--fast_mode', action='store_true', help='Только быстрые операции')
    parser.add_argument('--save_only', action='store_true', help='Только сохранить, не показывать')
    parser.add_argument('--create_test', action='store_true', help='Создать тестовое изображение')

    args = parser.parse_args()

    try:
        # Создание тестового изображения
        if args.create_test:
            test_file = create_test_image()
            if not args.image_path:
                args.image_path = test_file

        if not args.image_path:
            args.image_path = create_test_image()

        if not os.path.exists(args.image_path):
            print(f"❌ Файл не найден: {args.image_path}")
            return 1

        # Быстрая загрузка
        print(f"📷 Загрузка: {args.image_path}")
        start_load = time.time()

        with Image.open(args.image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            original_image = np.array(img)

        h, w, c = original_image.shape
        pixels = original_image.reshape(-1, c).astype(np.float32) / 255.0

        print(f"✅ Загружено за {time.time() - start_load:.2f}с: {len(pixels)} пикселей")

        # Быстрая кластеризация
        model = FastMultithreadClustering(
            n_clusters=args.clusters,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            n_threads=args.threads
        )

        # Обучение
        model.fit(pixels)

        # Быстрые метрики
        if not args.fast_mode:
            metrics = model.get_fast_metrics(pixels)
            print(f"📈 Silhouette Score: {metrics['silhouette_score']:.4f}")
            print(f"📈 Средняя уверенность: {metrics['avg_confidence']:.4f}")

        # Быстрая визуализация
        model.create_fast_visualization(
            original_image, pixels, (h, w),
            save_only=args.save_only
        )

        print("\n🎉 БЫСТРАЯ кластеризация завершена!")
        print("💡 Используйте --fast_mode для максимальной скорости")
        print("💡 Используйте --save_only чтобы избежать показа окон")

        return 0

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    print("🖥️  БЫСТРАЯ КЛАСТЕРИЗАЦИЯ:")
    print(f"   CPU ядер: {mp.cpu_count()}")
    print(f"   RAM: {psutil.virtual_memory().total / 1024**3:.1f} ГБ")
    print(f"   Numba: {'Да' if NUMBA_AVAILABLE else 'Нет'}")
    print(f"   Sklearn: {'Да' if SKLEARN_AVAILABLE else 'Нет'}")
    print("-" * 50)

    exit(main())
