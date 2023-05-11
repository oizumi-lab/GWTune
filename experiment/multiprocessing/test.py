# %%
from multiprocessing import Process, Queue

def my_function(input_data, output_queue):
    # この関数は、input_dataの各要素を2倍にしてoutput_queueに格納するものとします
    result = []
    for x in input_data:
        result.append(x * 2)
    output_queue.put(result)

if __name__ == '__main__':
    # テスト用の入力データ
    input_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # プロセス数
    num_processes = 4

    # 入力データを均等に分割して、各プロセスに割り当てる
    input_data_per_process = [input_data[i::num_processes] for i in range(num_processes)]

    # 出力結果を格納するキュー
    output_queue = Queue()

    # プロセスのリスト
    processes = []

    # プロセスを生成し、関数を実行する
    for i in range(num_processes):
        p = Process(target=my_function, args=(input_data_per_process[i], output_queue))
        p.start()
        processes.append(p)

    # 全プロセスの終了を待つ
    for p in processes:
        p.join()

    # 出力結果を取得する
    output_list = []
    while not output_queue.empty():
        output_list += output_queue.get()

    # 出力結果を表示する
    print(output_list)