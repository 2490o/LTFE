# 修改文件夹下的文件名，添加cp或者去除cp


# import os

# # 设置文件夹路径
# folder_path = "/data1/ysk/Div_code/Annotations"

# # 遍历文件夹下的所有文件
# for filename in os.listdir(folder_path):
#     # 检查是否是文件（不包括文件夹）
#     if os.path.isfile(os.path.join(folder_path, filename)):
#         # 拼接新的文件名
#         new_filename = "cp" + filename
#         # 重命名文件
#         os.rename(
#             os.path.join(folder_path, filename),
#             os.path.join(folder_path, new_filename)
#         )

# print("文件名已更新完成！")


# import os

# # 设置文件夹路径
# folder_path = "/data1/ysk/Div_code/Annotations"

# # 遍历文件夹下的所有文件
# for filename in os.listdir(folder_path):
#     # 检查文件是否以 "cp" 开头，并且是文件（不包括文件夹）
#     if filename.startswith("cp") and os.path.isfile(os.path.join(folder_path, filename)):
#         # 拼接新的文件名，去掉前面的 "cp"
#         new_filename = filename[2:]
#         # 重命名文件
#         os.rename(
#             os.path.join(folder_path, filename),
#             os.path.join(folder_path, new_filename)
#         )

# print("文件名前的 'cp' 已移除！")
