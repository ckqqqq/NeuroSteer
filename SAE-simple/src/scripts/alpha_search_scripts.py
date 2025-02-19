import argparse
import subprocess
import threading

'''
用于测试debate_test_0119.py脚本，通过修改alpha参数，观察opposerate和ppl的变化
期望：opposerate尽量提高（>50），ppl不要提升太高(<5)
'''



ALPHA_LIST = [i for i in range(25, 36)]

def main(alpha: int):
    print("alpha:", alpha)
    subprocess.run(['python', 'debate_test_0119.py', '--alpha', str(alpha)])

if __name__ == "__main__":
    '''
    并行执行两个，等两个执行完再执行另外两个
    '''
    for i in range(0, 6, 2):
        t1 = threading.Thread(target=main, args=(ALPHA_LIST[i],))
        t2 = threading.Thread(target=main, args=(ALPHA_LIST[i+1],))
        t1.start()
        print("==========================================")
        print(f"alpha: {ALPHA_LIST[i]} started")
        t2.start()
        print(f"alpha: {ALPHA_LIST[i+1]} started")
        t1.join()
        print("==========================================")
        print(f"alpha: {ALPHA_LIST[i]} done")
        t2.join()
        print("==========================================")
        print(f"alpha: {ALPHA_LIST[i+1]} done")
    print("All done!")