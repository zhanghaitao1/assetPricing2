if __name__ == '__main__':
    from multiprocessing import Pool
    p = Pool()
    result = p.map(task, args)