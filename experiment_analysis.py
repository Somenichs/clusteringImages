import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
from gradient_clustering import GradientImageClustering, ImageProcessor, MetricsCalculator
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import time
import os

class ExperimentAnalysis:
    """
    Класс для проведения экспериментального анализа
    """

    def __init__(self, output_dir='experiment_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

    def run_clustering_experiment(self, image_path, algorithms, n_clusters_range, runs=5):
        """
        Проведение эксперимента по кластеризации с различными алгоритмами
        """
        # Загрузка изображения
        image = ImageProcessor.load_image(image_path)
        pixels, shape = ImageProcessor.preprocess_image(image, normalize=True)

        image_name = os.path.basename(image_path)

        for n_clusters in n_clusters_range:
            for algorithm_name, algorithm_config in algorithms.items():

                algorithm_results = {
                    'image': image_name,
                    'algorithm': algorithm_name,
                    'n_clusters': n_clusters,
                    'times': [],
                    'silhouettes': [],
                    'db_indices': [],
                    'inertias': []
                }

                for run in range(runs):
                    start_time = time.time()

                    if algorithm_name == 'GradientDescent':
                        model = GradientImageClustering(**algorithm_config, n_clusters=n_clusters, random_state=run)
                        model.fit(pixels)
                        labels = model.labels
                        centroids = model.centroids

                    elif algorithm_name == 'KMeans':
                        model = KMeans(n_clusters=n_clusters, random_state=run, **algorithm_config)
                        labels = model.fit_predict(pixels)
                        centroids = model.cluster_centers_

                    elif algorithm_name == 'GaussianMixture':
                        model = GaussianMixture(n_components=n_clusters, random_state=run, **algorithm_config)
                        model.fit(pixels)
                        labels = model.predict(pixels)
                        centroids = model.means_

                    end_time = time.time()

                    # Вычисление метрик
                    if len(np.unique(labels)) > 1:
                        silhouette = MetricsCalculator.silhouette_score_safe(pixels, labels)
                        db_index = MetricsCalculator.davies_bouldin_index(pixels, labels, centroids)
                        inertia = MetricsCalculator.inertia(pixels, labels, centroids)
                    else:
                        silhouette = db_index = inertia = 0.0

                    algorithm_results['times'].append(end_time - start_time)
                    algorithm_results['silhouettes'].append(silhouette)
                    algorithm_results['db_indices'].append(db_index)
                    algorithm_results['inertias'].append(inertia)

                # Сохранение результатов
                result_summary = {
                    'image': image_name,
                    'algorithm': algorithm_name,
                    'n_clusters': n_clusters,
                    'mean_time': np.mean(algorithm_results['times']),
                    'std_time': np.std(algorithm_results['times']),
                    'mean_silhouette': np.mean(algorithm_results['silhouettes']),
                    'std_silhouette': np.std(algorithm_results['silhouettes']),
                    'mean_db_index': np.mean(algorithm_results['db_indices']),
                    'std_db_index': np.std(algorithm_results['db_indices']),
                    'mean_inertia': np.mean(algorithm_results['inertias']),
                    'std_inertia': np.std(algorithm_results['inertias'])
                }

                self.results.append(result_summary)

                print(f"{image_name} | {algorithm_name} | K={n_clusters} | "
                      f"Time: {result_summary['mean_time']:.3f}±{result_summary['std_time']:.3f}s | "
                      f"Silhouette: {result_summary['mean_silhouette']:.4f}")

    def generate_report(self):
        """
        Генерация отчета по экспериментам
        """
        df = pd.DataFrame(self.results)

        # Сохранение CSV
        csv_path = os.path.join(self.output_dir, 'experiment_results.csv')
        df.to_csv(csv_path, index=False)

        # Создание графиков
        self.plot_performance_comparison(df)
        self.plot_scalability_analysis(df)
        self.plot_quality_metrics(df)

        return df

    def plot_performance_comparison(self, df):
        """
        График сравнения производительности алгоритмов
        """
        plt.figure(figsize=(15, 5))

        # Время выполнения
        plt.subplot(1, 3, 1)
        sns.boxplot(data=df, x='algorithm', y='mean_time')
        plt.title('Время выполнения алгоритмов')
        plt.ylabel('Время (секунды)')
        plt.xticks(rotation=45)

        # Silhouette Score
        plt.subplot(1, 3, 2)
        sns.boxplot(data=df, x='algorithm', y='mean_silhouette')
        plt.title('Качество кластеризации (Silhouette Score)')
        plt.ylabel('Silhouette Score')
        plt.xticks(rotation=45)

        # Davies-Bouldin Index
        plt.subplot(1, 3, 3)
        sns.boxplot(data=df, x='algorithm', y='mean_db_index')
        plt.title('Davies-Bouldin Index (меньше = лучше)')
        plt.ylabel('DB Index')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_scalability_analysis(self, df):
        """
        Анализ масштабируемости по количеству кластеров
        """
        plt.figure(figsize=(12, 8))

        for algorithm in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == algorithm]

            plt.subplot(2, 2, 1)
            plt.plot(alg_data['n_clusters'], alg_data['mean_time'], 'o-', label=algorithm)
            plt.xlabel('Количество кластеров')
            plt.ylabel('Время выполнения (сек)')
            plt.title('Масштабируемость по времени')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.plot(alg_data['n_clusters'], alg_data['mean_silhouette'], 'o-', label=algorithm)
            plt.xlabel('Количество кластеров')
            plt.ylabel('Silhouette Score')
            plt.title('Качество vs Количество кластеров')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.plot(alg_data['n_clusters'], alg_data['mean_inertia'], 'o-', label=algorithm)
            plt.xlabel('Количество кластеров')
            plt.ylabel('Инерция')
            plt.title('Инерция vs Количество кластеров')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot(alg_data['n_clusters'], alg_data['mean_db_index'], 'o-', label=algorithm)
            plt.xlabel('Количество кластеров')
            plt.ylabel('Davies-Bouldin Index')
            plt.title('DB Index vs Количество кластеров')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'scalability_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_quality_metrics(self, df):
        """
        Детальный анализ метрик качества
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Корреляция между метриками
        metrics_df = df[['mean_silhouette', 'mean_db_index', 'mean_inertia', 'mean_time']]
        correlation_matrix = metrics_df.corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   ax=axes[0, 0], fmt='.3f')
        axes[0, 0].set_title('Корреляция между метриками')

        # Scatter plot: Silhouette vs DB Index
        for algorithm in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == algorithm]
            axes[0, 1].scatter(alg_data['mean_silhouette'], alg_data['mean_db_index'],
                             label=algorithm, alpha=0.7)

        axes[0, 1].set_xlabel('Silhouette Score')
        axes[0, 1].set_ylabel('Davies-Bouldin Index')
        axes[0, 1].set_title('Silhouette vs Davies-Bouldin')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Quality vs Time trade-off
        for algorithm in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == algorithm]
            axes[1, 0].scatter(alg_data['mean_time'], alg_data['mean_silhouette'],
                             label=algorithm, alpha=0.7)

        axes[1, 0].set_xlabel('Время выполнения (сек)')
        axes[1, 0].set_ylabel('Silhouette Score')
        axes[1, 0].set_title('Компромисс качество-скорость')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Статистика по алгоритмам
        algorithm_stats = df.groupby('algorithm').agg({
            'mean_silhouette': ['mean', 'std'],
            'mean_time': ['mean', 'std']
        }).round(4)

        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table_data = []
        for alg in algorithm_stats.index:
            row = [
                alg,
                f"{algorithm_stats.loc[alg, ('mean_silhouette', 'mean')]:.4f}±{algorithm_stats.loc[alg, ('mean_silhouette', 'std')]:.4f}",
                f"{algorithm_stats.loc[alg, ('mean_time', 'mean')]:.3f}±{algorithm_stats.loc[alg, ('mean_time', 'std')]:.3f}"
            ]
            table_data.append(row)

        table = axes[1, 1].table(cellText=table_data,
                               colLabels=['Алгоритм', 'Silhouette Score', 'Время (сек)'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Сводная статистика')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'quality_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()

def run_comprehensive_experiments():
    """
    Запуск комплексных экспериментов для дипломной работы
    """
    # Конфигурация алгоритмов
    algorithms = {
        'GradientDescent': {
            'learning_rate': 0.01,
            'max_iter': 100,
            'sigma': 1.0,
            'init_method': 'kmeans++'
        },
        'KMeans': {
            'n_init': 10,
            'max_iter': 300
        },
        'GaussianMixture': {
            'max_iter': 100,
            'n_init': 5
        }
    }

    # Создание синтетических изображений для тестирования
    test_images = []
    patterns = ['circles', 'rectangles', 'gradient']

    for pattern in patterns:
        img = create_synthetic_image(size=(150, 150), pattern=pattern)
        img_path = f'test_{pattern}.png'
        Image.fromarray(img).save(img_path)
        test_images.append(img_path)

    # Запуск экспериментов
    analyzer = ExperimentAnalysis()

    for img_path in test_images:
        print(f"\nЭксперименты с изображением: {img_path}")
        analyzer.run_clustering_experiment(
            img_path,
            algorithms,
            n_clusters_range=[2, 3, 4, 5],
            runs=5
        )

    # Генерация отчета
    results_df = analyzer.generate_report()

    print("\nЭксперименты завершены!")
    print(f"Результаты сохранены в папку: {analyzer.output_dir}")

    # Очистка временных файлов
    for img_path in test_images:
        if os.path.exists(img_path):
            os.remove(img_path)

    return results_df

if __name__ == "__main__":
    print("Запуск комплексного экспериментального анализа...")
    results = run_comprehensive_experiments()
    print("\nЛучшие результаты по Silhouette Score:")
    print(results.nlargest(5, 'mean_silhouette')[['image', 'algorithm', 'n_clusters', 'mean_silhouette', 'mean_time']])
