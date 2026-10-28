# -*- coding: utf-8 -*-
import io
import os
import sys
import argparse
from GoogleTM_Image_CatClassifier import CatImageClassifier
from MNISTClassifier import MNISTClassifier

if __name__ == "__main__":
    # 兼容低版本 Python 设置标准输出编码为 UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description='选择程序运行模式')
    group = parser.add_mutually_exclusive_group()

    group.add_argument('--gtm', action='store_true', help='启用 Google Teachable Machine 模式')
    group.add_argument('--mnist', action='store_true', help='启用手写数字识别模式')

    parser.add_argument('--static', action='store_true', help='使用静态图像预测模式')
    parser.add_argument('--realtime', action='store_true', help='使用实时预测模式')

    default_test_image_name = 'BritishShorthair01.jpg'
    parser.add_argument('--test_image_name', default=default_test_image_name, help='静态图像预测时的图像路径')
    parser.add_argument('--epoch', type=int, default=10, help='手写数字识别模式下的训练轮次')

    args = parser.parse_args()

    RED = '\033[31m'
    YELLOW = '\033[33m'
    RESET = '\033[0m'

    if (not args.gtm and (args.static or args.realtime or ('--test_image_name' in sys.argv))) or (not args.mnist and ('--epoch' in sys.argv)):
        if not args.mnist and ('--epoch' in sys.argv):
            print(f"{RED}\n错误：当未启用 手写数字识别 模式 (--mnist) 时，不能使用 --epoch 参数\n{RESET}")
        else:
            print(f"{RED}\n错误：当未启用 Google Teachable Machine 模式 (--gtm) 时，不能使用 --static、--realtime 或 --test_image_name 参数\n{RESET}")
        parser.print_help()
    elif args.gtm:
        """
        通过参数 --gtm 进入 Google Teachable Machine 模式
        """
        print("启用 Google Teachable Machine 模式")
        cat_classifier = CatImageClassifier()
        if args.static:
            # 预测 test 集里面的图像
            if args.realtime:
                print(f"{RED}\n错误：--static 和 --realtime 不能同时出现\n{RESET}")
                parser.print_help()
                exit(1)
            if '--test_image_name' not in sys.argv:
                print(f"{YELLOW}\n警告：\t您未指定测试图像名，现在使用默认图像进行测试{RESET}")
                print(f"{YELLOW}\t可使用 .\\run_gtm_static.sh -i <TEST_IMAGE_NAME> 来指定测试图像名\n{RESET}")
            print("对 测试集中的猫图 进行 静态预测")
            test_image_path = os.path.join('./Dataset/Cat/test/', args.test_image_name)
            cat_classifier.predict_image(test_image_path)
        elif args.realtime:
            # 预测实时摄像头的画面
            if '--test_image_name' in sys.argv:
                print(f"{RED}\n错误：--realtime 和 --test_image_name 不能同时出现\n{RESET}")
                parser.print_help()
                exit(1)
            print("对 摄像头中的影像 进行 实时预测")
            cat_classifier.real_time_prediction()
        else:
            print(f"{RED}\n错误：请指定 Google Teachable Machine 的运行模式 (--static 或 --realtime) \n{RESET}")
            parser.print_help()
    elif args.mnist:
        """
        通过参数 --mnist 进入手写数字识别模式
        """
        print("启用 手写数字识别 (MNIST)  模式")
        model = MNISTClassifier().to(MNISTClassifier.device)
        best_loss = 10000
        best_acc = 0
        if '--epoch' not in sys.argv:
            print(f"{YELLOW}\n警告：\t您未指定训练轮次，现在使用默认的训练轮次 (epoch = 10) 进行训练{RESET}")
            print(f"{YELLOW}\t可使用 .\\run_mnist.sh -e <EPOCH_NUM> 来指定训练轮次\n{RESET}")
        if args.epoch <= 0:
            print(f"{RED}\n错误：请指定 Google Teachable Machine 的运行模式 (--static 或 --realtime) \n{RESET}")
        for epoch in range(args.epoch):
            model.mnist_train(epoch)
            temp_loss, temp_acc = model.mnist_eval()
            best_loss = min(temp_loss, best_loss)
            best_acc = max(temp_acc, best_acc)
            print("")

        print("训练结束，最佳结果如下：")
        print(f"最佳损失: {best_loss:.6f}")
        print(f"最佳准确率: {best_acc * 100:.2f}%")
        print("")
    else:
        print(f"{RED}\n错误：请指定运行模式\n{RESET}")
        parser.print_help()
