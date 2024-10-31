import logging

def setup_logging(filename):
    # 创建 logger
    logger = logging.getLogger('Pathplanning')
    logger.setLevel(logging.DEBUG)  # 设置最低的日志级别

    # 创建文件处理器，记录所有的日志到文件
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)

    # 创建控制台处理器，只记录警告以上级别的日志到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger