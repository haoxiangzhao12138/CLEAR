import os

def compare_folders(folder1, folder2):
    # 获取两个文件夹中的文件名集合
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    if files1 == files2:
        print("两个文件夹中的文件名完全一致")
    else:
        print("两个文件夹中的文件名不一致")

        only_in_1 = files1 - files2
        only_in_2 = files2 - files1

        if only_in_1:
            print("\n只在文件夹1中存在的文件:")
            for f in sorted(only_in_1):
                print(f)

        if only_in_2:
            print("\n只在文件夹2中存在的文件:")
            for f in sorted(only_in_2):
                print(f)


if __name__ == "__main__":
    folder1 = "/root/CLEAR/datasets/processed_dataset/rl/corruption_images"
    folder2 = "/root/CLEAR/datasets/processed_dataset/rl/images"

    compare_folders(folder1, folder2)