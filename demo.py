import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
from gradient_clustering import GradientImageClustering, ImageProcessor, MetricsCalculator, Visualizer
from sklearn.cluster import KMeans
import time

def create_synthetic_image(size=(200, 200), pattern='circles'):
    """
    Создание синтетических изображений для тестирования
    """
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)

    if pattern == 'circles':
        # Три цветных круга
        draw.ellipse([20, 20, 80, 80], fill='red')
        draw.ellipse([60, 60, 120, 120], fill='blue')
        draw.ellipse([100, 100, 160, 160], fill='green')

    elif pattern == 'rectangles':
        # Цветные прямоугольники
        draw.rectangle([10, 10, 90, 90], fill='red')
        draw.rectangle([110, 10, 190, 90], fill='blue')
        draw.rectangle([60, 110, 140, 190], fill='green')

    elif pattern == 'gradient':
        # Градиентный переход
        width, height = size
        for x in range(width):
            color_val = int(255 * x / width)
            draw.line([(x, 0), (x, height)], fill=(color_val, 0, 255-color_val))

    return np.array(img)

def benchmark_algorithms(image, n_clusters=3, runs=3):
    """
    Сравнительное тестирование алгоритмов
    """
    pixels, shape = ImageProcessor.preprocess_image(image, normalize=True)

    results = {
        'Gradient Descent': {'times': [], 'silhouettes': [], 'db_indices': []},
        'K-means': {'times': [], 'silhouettes': [], 'db_indices': []}
    }

    print(f"Тестирование на изображении размером {shape}")
    print(f"Количество пикселей: {len(pixels)}")
    print(f"Количество кластеров: {n_clusters}")
    print(f"Количество запусков: {runs}")
    print("="*50)

    # Тестирование градиентного спуска
    print("Тестирование градиентного спуска...")
    for run in range(runs):
        start_time = time.time()

        model = GradientImageClustering(
            n_clusters=n_clusters,
            learning_rate=0.01,
            max_iter=100,
            random_state=run
        )
        model.fit(pixels)

        end_time = time.time()

        # Метрики
        silhouette = MetricsCalculator.silhouette_score_safe(pixels, model.labels)
        db_index = MetricsCalculator.davies_bouldin_index(pixels, model.labels, model.centroids)

        results['Gradient Descent']['times'].append(end_time - start_time)
        results['Gradient Descent']['silhouettes'].append(silhouette)
        results['Gradient Descent']['db_indices'].append(db_index)

        print(f"  Запуск {run+1}: Время={end_time-start_time:.2f}с, Silhouette={silhouette:.4f}")

    # Тестирование K-means
    print("\nТестирование K-means...")
    for run in range(runs):
        start_time = time.time()

        kmeans = KMeans(n_clusters=n_clusters, random_state=run, n_init=10)
        labels = kmeans.fit_predict(pixels)

        end_time = time.time()

        # Метрики
        silhouette = MetricsCalculator.silhouette_score_safe(pixels, labels)
        db_index = MetricsCalculator.davies_bouldin_index(pixels, labels, kmeans.cluster_centers_)

        results['K-means']['times'].append(end_time - start_time)
        results['K-means']['silhouettes'].append(silhouette)
        results['K-means']['db_indices'].append(db_index)

        print(f"  Запуск {run+1}: Время={end_time-start_time:.2f}с, Silhouette={silhouette:.4f}")

    return results

def print_benchmark_results(results):
    """
    Вывод результатов бенчмарка
    """
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ СРАВНИТЕЛЬНОГО ТЕСТИРОВАНИЯ")
    print("="*60)

    for algorithm, metrics in results.items():
        print(f"\n{algorithm}:")
        print(f"  Среднее время выполнения: {np.mean(metrics['times']):.3f} ± {np.std(metrics['times']):.3f} сек")
        print(f"  Средний Silhouette Score: {np.mean(metrics['silhouettes']):.4f} ± {np.std(metrics['silhouettes']):.4f}")
        print(f"  Средний Davies-Bouldin Index: {np.mean(metrics['db_indices']):.4f} ± {np.std(metrics['db_indices']):.4f}")

def demo_parameter_sensitivity():
    """
    Демонстрация чувствительности к параметрам
    """
    print("\nДемонстрация чувствительности к параметрам...")

    # Создаем тестовое изображение
    image = create_synthetic_image(pattern='circles')
    pixels, shape = ImageProcessor.preprocess_image(image, normalize=True)

    # Тестируем разные значения sigma
    sigma_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = []

    for sigma in sigma_values:
        model = GradientImageClustering(
            n_clusters=3,
            learning_rate=0.01,
            max_iter=50,
            sigma=sigma,
            random_state=42
        )
        model.fit(pixels)

        silhouette = MetricsCalculator.silhouette_score_safe(pixels, model.labels)
        results.append(silhouette)
        print(f"  Sigma = {sigma}: Silhouette = {silhouette:.4f}")

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_values, results, 'bo-')
    plt.xlabel('Значение параметра σ')
    plt.ylabel('Silhouette Score')
    plt.title('Влияние параметра σ на качество кластеризации')
    plt.grid(True)
    plt.show()

def main():
    """
    Основная демонстрационная функция
    """
    print("ДЕМОНСТРАЦИЯ КЛАСТЕРИЗАЦИИ ИЗОБРАЖЕНИЙ МЕТОДОМ ГРАДИЕНТНОГО СПУСКА")
    print("="*70)

    # 1. Создание синтетических изображений
    print("\n1. Создание тестовых изображений...")

    test_images = {
        'circles': create_synthetic_image(pattern='circles'),
        'rectangles': create_synthetic_image(pattern='rectangles'),
        'gradient': create_synthetic_image(pattern='gradient')
    }

    # Визуализация тестовых изображений
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (name, img) in enumerate(test_images.items()):
        axes[i].imshow(img)
        axes[i].set_title(f'Тестовое изображение: {name}')
        axes[i].axis('off')
    plt.suptitle('Синтетические изображения для тестирования')
    plt.tight_layout()
    plt.show()

    # 2. Базовая демонстрация алгоритма
    print("\n2. Базовая демонстрация алгоритма...")

    test_image = test_images['circles']
    pixels, shape = ImageProcessor.preprocess_image(test_image, normalize=True)

    model = GradientImageClustering(
        n_clusters=3,
        learning_rate=0.01,
        max_iter=50,
        sigma=1.0,
        random_state=42
    )

    model.fit(pixels)

    # Восстановление изображения
    clustered_image = ImageProcessor.reconstruct_image(
        pixels, shape, model.labels, model.centroids
    )

    # Вычисление карты уверенности
    confidence_map = model.get_confidence_map(pixels)

    # Визуализация
    Visualizer.plot_results(
        test_image, clustered_image,
        model.centroids * 255, model.labels,
        model.loss_history, confidence_map, shape
    )

    # 3. Сравнительное тестирование
    print("\n3. Сравнительное тестирование алгоритмов...")

    benchmark_results = benchmark_algorithms(test_image, n_clusters=3, runs=3)
    print_benchmark_results(benchmark_results)

    # 4. Анализ чувствительности к параметрам
    print("\n4. Анализ чувствительности к параметрам...")
    demo_parameter_sensitivity()

    # 5. Тестирование на изображениях разного типа
    print("\n5. Тестирование на различных типах изображений...")

    for name, img in test_images.items():
        print(f"\nТестирование на изображении '{name}':")

        pixels, shape = ImageProcessor.preprocess_image(img, normalize=True)

        model = GradientImageClustering(
            n_clusters=3,
            learning_rate=0.01,
            max_iter=30,
            random_state=42
        )

        start_time = time.time()
        model.fit(pixels)
        end_time = time.time()

        silhouette = MetricsCalculator.silhouette_score_safe(pixels, model.labels)

        print(f"  Время выполнения: {end_time - start_time:.2f} секунд")
        print(f"  Количество итераций: {model.n_iter}")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Сходимость: {'Да' if model.converged else 'Нет'}")

    print("\nДемонстрация завершена!")

if __name__ == "__main__":
    main()
