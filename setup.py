import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Выполнение команды с обработкой ошибок"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - успешно!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при {description}:")
        print(f"   {e.stderr}")
        return False

def create_requirements():
    """Создание файла requirements.txt"""
    requirements = """numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=8.3.0
scikit-learn>=1.0.0
pandas>=1.3.0
seaborn>=0.11.0
numba>=0.56.0
psutil>=5.8.0
argparse
"""
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✅ Файл requirements.txt создан")

def create_test_image():
    """Создание тестового изображения"""
    try:
        from PIL import Image, ImageDraw
        
        # Создаем изображение с тремя цветными кругами
        img = Image.new('RGB', (300, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # Рисуем круги разных цветов
        draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0))    # Красный
        draw.ellipse([100, 100, 200, 200], fill=(0, 0, 255))  # Синий
        draw.ellipse([150, 150, 250, 250], fill=(0, 255, 0))  # Зеленый
        
        img.save('test_image.png')
        print("✅ Тестовое изображение создано: test_image.png")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания тестового изображения: {e}")
        return False

def create_gradient_clustering_module():
    """Создание основного модуля (заглушка - нужно вручную скопировать код)"""
    stub_code = '''#!/usr/bin/env python3
"""
ВНИМАНИЕ: Это заглушка!
Скопируйте сюда полный код из артефакта gradient_clustering.py
"""

print("❌ Необходимо скопировать код основного модуля!")
print("📋 Скопируйте код из артефакта 'gradient_clustering.py' в этот файл")
'''
    
    if not os.path.exists('gradient_clustering.py'):
        with open('gradient_clustering.py', 'w') as f:
            f.write(stub_code)
        print("⚠️  Создан файл-заглушка gradient_clustering.py")
        print("📋 ВАЖНО: Скопируйте в него код из артефакта!")
        return False
    return True

def create_demo_module():
    """Создание демонстрационного модуля (заглушка)"""
    stub_code = '''#!/usr/bin/env python3
"""
ВНИМАНИЕ: Это заглушка!
Скопируйте сюда код demo.py из второго артефакта
"""

print("❌ Необходимо скопировать код демонстрационного модуля!")
print("📋 Скопируйте код из артефакта 'demo.py'")
'''
    
    if not os.path.exists('demo.py'):
        with open('demo.py', 'w') as f:
            f.write(stub_code)
        print("⚠️  Создан файл-заглушка demo.py")
        return False
    return True

def test_installation():
    """Тестирование установки"""
    test_code = """
import sys
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from sklearn.cluster import KMeans
    import pandas as pd
    import seaborn as sns
    
    print("✅ Все основные библиотеки импортированы успешно!")
    print(f"✅ Python версия: {sys.version}")
    print(f"✅ NumPy версия: {np.__version__}")
    print(f"✅ Matplotlib версия: {plt.matplotlib.__version__}")
    
    # Тест создания изображения
    test_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    Image.fromarray(test_img).save('installation_test.png')
    print("✅ Тест создания изображения прошел успешно!")
    
    import os
    os.remove('installation_test.png')
    
    exit(0)
    
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Ошибка тестирования: {e}")
    exit(1)
"""
    
    try:
        exec(test_code)
        return True
    except SystemExit as e:
        return e.code == 0

def main():
    """Основная функция установки"""
    print("🚀 АВТОМАТИЧЕСКАЯ УСТАНОВКА ПРОЕКТА КЛАСТЕРИЗАЦИИ ИЗОБРАЖЕНИЙ")
    print("=" * 60)
    
    # 1. Проверка Python версии
    if sys.version_info < (3, 7):
        print("❌ Требуется Python 3.7 или выше!")
        print(f"   Текущая версия: {sys.version}")
        return False
    
    print(f"✅ Python версия: {sys.version}")
    
    # 2. Создание файлов проекта
    print("\n📁 Создание файлов проекта...")
    create_requirements()
    create_gradient_clustering_module()
    create_demo_module()
    
    # 3. Установка зависимостей
    print("\n📦 Установка зависимостей...")
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Обновление pip"):
        print("⚠️  Предупреждение: не удалось обновить pip")
    
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Установка зависимостей"):
        print("❌ Не удалось установить зависимости!")
        print("🔧 Попробуйте установить вручную:")
        print("   pip install --user -r requirements.txt")
        return False
    
    # 4. Тестирование установки
    print("\n🧪 Тестирование установки...")
    if test_installation():
        print("✅ Тестирование прошло успешно!")
    else:
        print("❌ Ошибка при тестировании!")
        return False
    
    # 5. Создание тестового изображения
    print("\n🎨 Создание тестового изображения...")
    create_test_image()
    
    # 6. Создание папки для результатов
    os.makedirs('experiment_results', exist_ok=True)
    print("✅ Папка experiment_results создана")
    
    # 7. Создание файла с инструкциями
    instructions = """
# 🎓 ИНСТРУКЦИИ ПО ИСПОЛЬЗОВАНИЮ

## ⚠️ ВАЖНО: Завершите установку!

1. Скопируйте код из артефакта 'gradient_clustering.py' в файл gradient_clustering.py
2. Скопируйте код из артефакта 'demo.py' в файл demo.py
3. Скопируйте код из артефакта 'experiment_analysis.py' в файл experiment_analysis.py

## 🚀 Быстрый старт:

```bash
# Базовая кластеризация
python gradient_clustering.py test_image.png

# С параметрами
python gradient_clustering.py test_image.png --clusters 5 --compare_kmeans

# Демонстрация
python demo.py
```

## 📞 Решение проблем:

Если что-то не работает:
1. Проверьте, что все файлы скопированы из артефактов
2. Убедитесь, что все библиотеки установлены: pip install -r requirements.txt
3. Проверьте версию Python: должна быть 3.7+
"""
    
    with open('INSTRUCTIONS.md', 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("\n🎉 УСТАНОВКА ЗАВЕРШЕНА!")
    print("=" * 40)
    print("✅ Файлы проекта созданы")
    print("✅ Зависимости установлены") 
    print("✅ Тестовое изображение создано")
    print("✅ Инструкции сохранены в INSTRUCTIONS.md")
    
    print("\n⚠️  ВАЖНО: Завершите установку!")
    print("1. Скопируйте код из артефактов Claude в файлы:")
    print("   - gradient_clustering.py")  
    print("   - demo.py")
    print("   - experiment_analysis.py")
    
    print("\n🚀 После этого можете запускать:")
    print("   python gradient_clustering.py test_image.png")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
