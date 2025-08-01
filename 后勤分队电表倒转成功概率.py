import enum
import multiprocessing
from collections.abc import Iterable
from enum import Enum
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg
import scipy.stats


def 计算投出至少若干枚特定类型通宝的概率(*, 钱盒容量: int, 特定类型通宝的数量: int, 投钱个数: int, 需求数量: int) -> float:
    rv = scipy.stats.hypergeom(M=钱盒容量, n=特定类型通宝的数量, N=投钱个数)
    prob = rv.sf(需求数量 - 1)

    return prob


后勤分队钱盒容量 = 10
后勤分队投钱个数 = 3

投出茧成绢的概率 = 计算投出至少若干枚特定类型通宝的概率(钱盒容量=后勤分队钱盒容量, 特定类型通宝的数量=1, 投钱个数=后勤分队投钱个数, 需求数量=1)


def 增加1次源石锭(初始源石锭数量: int) -> int:
    return 初始源石锭数量 + min(初始源石锭数量 // 4, 99)


type 状态类 = 过渡态类 | 吸收态类


class 过渡态类(NamedTuple):
    源石锭数量: int
    烛火数量: int
    票券可投钱次数: int


class 吸收态类(Enum):
    源石锭达到阈值 = enum.auto()


class 求解器类:
    def __init__(self, *, 电表倒转成功源石锭阈值: int, 厉钱数量: int, 每次投钱消耗票券数量: int):
        self.电表倒转成功源石锭阈值: int = 电表倒转成功源石锭阈值
        self.厉钱数量: int = 厉钱数量
        self.每次投钱消耗票券数量: int = 每次投钱消耗票券数量

        self.投出至少2枚厉钱的概率 = 计算投出至少若干枚特定类型通宝的概率(钱盒容量=后勤分队钱盒容量, 特定类型通宝的数量=厉钱数量, 投钱个数=后勤分队投钱个数, 需求数量=2)
        self.投出至少2枚厉钱所需投钱次数的分布: scipy.stats.rv_discrete_frozen = scipy.stats.geom(p=self.投出至少2枚厉钱的概率)  # type: ignore

    def 获取状态(self, *, 源石锭数量: int, 烛火数量: int, 票券可投钱次数: int) -> 状态类:
        if 源石锭数量 >= self.电表倒转成功源石锭阈值:
            return 吸收态类.源石锭达到阈值
        else:
            return 过渡态类(源石锭数量=源石锭数量, 烛火数量=烛火数量, 票券可投钱次数=票券可投钱次数)

    def 状态转移(self, 起始状态: 状态类) -> list[tuple[状态类, float]]:
        转移概率列表: list[tuple[状态类, float]] = []

        if isinstance(起始状态, 吸收态类):  # 已经成功
            转移概率列表.append((起始状态, 1))

        else:
            起始源石锭数量 = 起始状态.源石锭数量
            起始烛火数量 = 起始状态.烛火数量
            起始票券可投钱次数 = 起始状态.票券可投钱次数

            if 起始源石锭数量 < 20:
                每次尝试消耗源石锭数量 = 10
                最多可尝试次数 = 起始源石锭数量 // 每次尝试消耗源石锭数量
                尝试成功获得烛火数量 = 2
                尝试成功获得票券可投钱次数 = 4 // self.每次投钱消耗票券数量
            elif 起始源石锭数量 < 50:
                每次尝试消耗源石锭数量 = 20
                最多可尝试次数 = 起始源石锭数量 // 每次尝试消耗源石锭数量
                尝试成功获得烛火数量 = 3
                尝试成功获得票券可投钱次数 = 6 // self.每次投钱消耗票券数量
            else:
                每次尝试消耗源石锭数量 = 50
                最多可尝试次数 = 起始源石锭数量 // 每次尝试消耗源石锭数量
                尝试成功获得烛火数量 = 6
                尝试成功获得票券可投钱次数 = 12 // self.每次投钱消耗票券数量

            if 起始烛火数量 <= 0:  # 没有烛火，失败
                转移概率列表.append((起始状态, 1))

            elif 起始票券可投钱次数 > 0:  # 有票券，则投钱
                转移概率列表.extend([
                    (self.获取状态(源石锭数量=增加1次源石锭(起始源石锭数量), 烛火数量=起始烛火数量, 票券可投钱次数=起始票券可投钱次数 - 1), 投出茧成绢的概率),  # 投出茧成绢
                    (self.获取状态(源石锭数量=起始源石锭数量, 烛火数量=起始烛火数量, 票券可投钱次数=起始票券可投钱次数 - 1), 1 - 投出茧成绢的概率)  # 没有投出茧成绢
                ])

            elif 起始源石锭数量 < 10:  # 没有源石锭，失败
                转移概率列表.append((起始状态, 1))

            elif 起始烛火数量 > 1:  # 有烛火，则拿票券
                for 尝试次数 in range(1, 最多可尝试次数 + 1):
                    概率 = self.投出至少2枚厉钱所需投钱次数的分布.pmf(尝试次数)
                    转移概率列表.append((self.获取状态(源石锭数量=起始源石锭数量 - 每次尝试消耗源石锭数量 * 尝试次数, 烛火数量=起始烛火数量 - 1, 票券可投钱次数=起始票券可投钱次数 + 尝试成功获得票券可投钱次数), 概率))  # 若干次投钱后投出了2枚厉钱
                失败概率 = self.投出至少2枚厉钱所需投钱次数的分布.sf(最多可尝试次数)
                转移概率列表.append((self.获取状态(源石锭数量=起始源石锭数量 - 每次尝试消耗源石锭数量 * 最多可尝试次数, 烛火数量=起始烛火数量 - 1, 票券可投钱次数=起始票券可投钱次数), 失败概率))  # 没有投出2枚厉钱

            else:  # 没有烛火，则拿烛火
                for 尝试次数 in range(1, 最多可尝试次数 + 1):
                    概率 = self.投出至少2枚厉钱所需投钱次数的分布.pmf(尝试次数)
                    转移概率列表.append((self.获取状态(源石锭数量=起始源石锭数量 - 每次尝试消耗源石锭数量 * 尝试次数, 烛火数量=起始烛火数量 - 1 + 尝试成功获得烛火数量, 票券可投钱次数=起始票券可投钱次数), 概率))  # 若干次投钱后投出了2枚厉钱
                失败概率 = self.投出至少2枚厉钱所需投钱次数的分布.sf(最多可尝试次数)
                转移概率列表.append((self.获取状态(源石锭数量=起始源石锭数量 - 每次尝试消耗源石锭数量 * 最多可尝试次数, 烛火数量=起始烛火数量 - 1, 票券可投钱次数=起始票券可投钱次数), 失败概率))  # 没有投出2枚厉钱

        转移概率列表 = [(目标状态, 概率) for 目标状态, 概率 in 转移概率列表 if 概率 > 0]
        assert np.isclose(sum(x[1] for x in 转移概率列表), 1)
        return 转移概率列表

    def 构建状态列表(self) -> None:
        self.状态列表: list[状态类] = []
        self.状态列表.extend(过渡态类(源石锭数量=源石锭数量, 烛火数量=烛火数量, 票券可投钱次数=票券可投钱次数)
                         for 源石锭数量 in range(0, self.电表倒转成功源石锭阈值 + 1, 1)
                         for 烛火数量 in range(0, 6 + 1, 1)
                         for 票券可投钱次数 in range(0, 12 // self.每次投钱消耗票券数量 + 1, 1))
        self.状态列表.append(吸收态类.源石锭达到阈值)

        self.状态数量: int = len(self.状态列表)
        self.状态索引: dict[状态类, int] = {状态: i for i, 状态 in enumerate(self.状态列表)}

        self.失败状态序号列表: list[int] = [状态序号 for 状态序号, 状态 in enumerate(self.状态列表) if not isinstance(状态, 吸收态类) and (状态.烛火数量 == 0 or (状态.源石锭数量 < 10 and 状态.票券可投钱次数 == 0))]
        self.成功状态序号列表: list[int] = [self.状态索引[吸收态类.源石锭达到阈值]]
        self.中间态序号列表: list[int] = [状态序号 for 状态序号, 状态 in enumerate(self.状态列表) if 状态序号 not in self.失败状态序号列表 and 状态序号 not in self.成功状态序号列表]

    def 构建状态转移矩阵(self) -> None:
        dok_状态转移矩阵 = scipy.sparse.dok_array((self.状态数量, self.状态数量))
        for 起始状态序号, 起始状态 in enumerate(self.状态列表):
            转移概率列表 = self.状态转移(起始状态)
            for 目标状态, 概率 in 转移概率列表:
                目标状态序号 = self.状态索引[目标状态]
                dok_状态转移矩阵[起始状态序号, 目标状态序号] += 概率
        self.状态转移矩阵 = dok_状态转移矩阵.tocsr()

    # def 迭代计算成功概率(self, 初始源石锭数量: int, 迭代次数: int) -> tuple[float, float, float]:
    #     初始状态 = 过渡态类(源石锭数量=初始源石锭数量, 烛火数量=1, 票券可投钱次数=0)
    #     当前状态分布 = np.zeros(self.状态数量)
    #     当前状态分布[self.状态索引[初始状态]] = 1

    #     for _ in range(迭代次数):
    #         当前状态分布 = 当前状态分布 @ self.状态转移矩阵

    #     失败概率 = np.sum(当前状态分布[self.失败状态序号列表])
    #     成功概率 = np.sum(当前状态分布[self.成功状态序号列表])
    #     中间态概率 = np.sum(当前状态分布[self.中间态序号列表])

    #     return 失败概率, 成功概率, 中间态概率

    def 计算成功概率(self) -> list[tuple[int, int, int, int, int, int, float]]:
        失败状态矩阵 = scipy.sparse.dok_array((len(self.失败状态序号列表), self.状态数量))
        for i, 状态序号 in enumerate(self.失败状态序号列表):
            失败状态矩阵[i, 状态序号] = 1
        失败状态矩阵 = 失败状态矩阵.tocsr()

        成功状态矩阵 = scipy.sparse.dok_array((len(self.成功状态序号列表), self.状态数量))
        for i, 状态序号 in enumerate(self.成功状态序号列表):
            成功状态矩阵[i, 状态序号] = 1
        成功状态矩阵 = 成功状态矩阵.tocsr()

        矩阵 = scipy.sparse.vstack([self.状态转移矩阵 - scipy.sparse.identity(self.状态数量, format='csr'), 失败状态矩阵, 成功状态矩阵])
        向量 = np.hstack([np.zeros(self.状态数量), np.zeros(len(self.失败状态序号列表)), np.ones(len(self.成功状态序号列表))])

        最小二乘结果 = scipy.sparse.linalg.lsqr(矩阵, 向量, atol=0, btol=0)
        # print(最小二乘结果)
        最小二乘解 = 最小二乘结果[0]

        结果列表 = []
        for 初始源石锭数量 in range(0, self.电表倒转成功源石锭阈值 + 1, 1):
            for 初始烛火数量 in range(1, 6 + 1, 1):
                for 初始票券可投钱次数 in range(0, 12 // self.每次投钱消耗票券数量 + 1, 1):
                    状态 = 过渡态类(源石锭数量=初始源石锭数量, 烛火数量=初始烛火数量, 票券可投钱次数=初始票券可投钱次数)
                    状态序号 = self.状态索引[状态]
                    成功概率 = 最小二乘解[状态序号]
                    结果列表.append((self.电表倒转成功源石锭阈值, self.厉钱数量, self.每次投钱消耗票券数量, 初始源石锭数量, 初始烛火数量, 初始票券可投钱次数, 成功概率))
        return 结果列表


def 单个厉钱数量计算(电表倒转成功源石锭阈值: int, 厉钱数量: int, 每次投钱消耗票券数量: int) -> list[tuple[int, int, int, int, int, int, float]]:
    求解器 = 求解器类(电表倒转成功源石锭阈值=电表倒转成功源石锭阈值, 厉钱数量=厉钱数量, 每次投钱消耗票券数量=每次投钱消耗票券数量)
    求解器.构建状态列表()
    求解器.构建状态转移矩阵()
    print(f"电表倒转成功源石锭阈值：{电表倒转成功源石锭阈值}，厉钱数量：{厉钱数量}，每次投钱消耗票券数量：{每次投钱消耗票券数量}，状态转移矩阵构建完成，状态数量：{求解器.状态数量}")
    结果列表 = 求解器.计算成功概率()
    print(f"电表倒转成功源石锭阈值：{电表倒转成功源石锭阈值}，厉钱数量：{厉钱数量}，每次投钱消耗票券数量：{每次投钱消耗票券数量}，最小二乘解计算完成")
    return 结果列表


if __name__ == "__main__":
    电表倒转成功源石锭阈值 = 4096
    待计算的厉钱数量和每次投钱消耗票券数量范围 = [
        (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1),
        (5, 2), (6, 2), (7, 2), (8, 2), (9, 2)
    ]

    with multiprocessing.Pool(processes=len(待计算的厉钱数量和每次投钱消耗票券数量范围)) as pool:
        results = pool.starmap(
            单个厉钱数量计算,
            ((电表倒转成功源石锭阈值, 厉钱数量, 每次投钱消耗票券数量)
             for 厉钱数量, 每次投钱消耗票券数量 in 待计算的厉钱数量和每次投钱消耗票券数量范围)
        )

    结果列表 = []
    for 单个结果 in results:
        结果列表.extend(单个结果)

    df = pd.DataFrame(结果列表, columns=["电表倒转成功源石锭阈值", "厉钱数量", "每次投钱消耗票券数量", "初始源石锭数量", "初始烛火数量", "初始票券可投钱次数", "成功概率"])
    df.to_csv("后勤分队电表倒转成功概率.csv.gz", index=False, compression="gzip")
