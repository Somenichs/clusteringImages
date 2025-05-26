import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

print("✅ Все библиотеки установлены успешно!")
print(f"NumPy версия: {np.__version__}")
print(f"Matplotlib версия: {plt.matplotlib.__version__}")

# Создаем тестовое изображение
test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
Image.fromarray(test_image).save('test_image.png')
print("✅ Тестовое изображение создано: test_image.png")
