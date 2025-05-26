from PIL import Image, ImageDraw
import numpy as np

def create_test_image():
    # Создаем изображение 300x300
    img = Image.new('RGB', (300, 300), color='white')
    draw = ImageDraw.Draw(img)

    # Рисуем цветные круги
    draw.ellipse([50, 50, 150, 150], fill='red')
    draw.ellipse([100, 100, 200, 200], fill='blue')
    draw.ellipse([150, 150, 250, 250], fill='green')

    img.save('test_circles.png')
    print("✅ Тестовое изображение создано: test_circles.png")

if __name__ == "__main__":
    create_test_image()
