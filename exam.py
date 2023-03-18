def solution(n, m, section):
    wall = [1] * n
    clean_cnt = 0
    for idx in range(1,len(wall)+1) :
        if idx in section:
            wall[idx-1] = 0

    for j in range(n):
        if wall[j] == 0:
            clean_cnt = clean_cnt + 1
            wall[j:j+(m-1)] = [1] * (m-1)
        
    return clean_cnt

def solution1(n, m, section):
    answer = 0
    # N 미터 벽 
    # 페인트가 벗겨져 칠하기로함
    # 룰러의 길이 M
    #   - 구역의 일부부만 칠하기 안됨

    # Greedy

    coloredWall = 0
    for wall in section:
        if wall > coloredWall:
            coloredWall = wall + m -1
            answer += 1
    return answer

from collections import deque
def solution2(n, m, section):
    cnt = 0
    q = deque(section)
    while q:
        v = q.popleft()
        cnt += 1
        while q and q[0] < v+m:
            q.popleft()
    return cnt

from math import sqrt,pow
def solution3(k,d):
    cnt = 0
    dist = 0 
    for i in range(1000000):
        if k*i < d:
            cnt += 1
        else :
            break
    for i in range(1000000):
        if k*i < d:
            cnt += 1
        else :
            break
    for i in range(1000000):
        dist = int(sqrt(pow(k*i,2)))
        if dist < d :
            cnt += 1
        else:
            break
    return cnt

if __name__ == "__main__" :
    solution3(2,4)
