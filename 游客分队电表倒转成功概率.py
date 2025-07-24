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


游客分队钱盒容量 = 12
游客分队投钱个数 = 5

投出茧成绢的概率 = 计算投出至少若干枚特定类型通宝的概率(钱盒容量=游客分队钱盒容量, 特定类型通宝的数量=1, 投钱个数=游客分队投钱个数, 需求数量=1)


def 增加1次源石锭(初始源石锭数量: int) -> int:
    return 初始源石锭数量 + min(初始源石锭数量 // 4, 99)


type 状态类 = 过渡态类 | 吸收态类


class 过渡态类(NamedTuple):
    源石锭数量: int
    烛火数量: int


class 吸收态类(Enum):
    源石锭达到阈值 = enum.auto()


class 求解器类:
    def __init__(self, *, 电表倒转成功源石锭阈值: int, 衡钱数量: int, 厉钱数量: int):
        self.电表倒转成功源石锭阈值: int = 电表倒转成功源石锭阈值
        self.衡钱数量: int = 衡钱数量
        self.厉钱数量: int = 厉钱数量

        self.投出至少1枚衡钱的概率 = 计算投出至少若干枚特定类型通宝的概率(钱盒容量=游客分队钱盒容量, 特定类型通宝的数量=衡钱数量, 投钱个数=游客分队投钱个数, 需求数量=1)
        self.投出至少1枚衡钱所需投钱次数的分布: scipy.stats.rv_discrete_frozen = scipy.stats.geom(p=self.投出至少1枚衡钱的概率)  # type: ignore

        self.投出至少2枚厉钱的概率 = 计算投出至少若干枚特定类型通宝的概率(钱盒容量=游客分队钱盒容量, 特定类型通宝的数量=厉钱数量, 投钱个数=游客分队投钱个数, 需求数量=2)
        self.投出至少2枚厉钱所需投钱次数的分布: scipy.stats.rv_discrete_frozen = scipy.stats.geom(p=self.投出至少2枚厉钱的概率)  # type: ignore

    def 获取状态(self, *, 源石锭数量: int, 烛火数量: int) -> 状态类:
        if 源石锭数量 >= self.电表倒转成功源石锭阈值:
            return 吸收态类.源石锭达到阈值
        else:
            return 过渡态类(源石锭数量=源石锭数量, 烛火数量=烛火数量)

    def 状态转移(self, 起始状态: 状态类) -> list[tuple[状态类, float]]:
        转移概率列表: list[tuple[状态类, float]] = []

        if isinstance(起始状态, 吸收态类):  # 已经成功
            转移概率列表.append((起始状态, 1))

        else:
            起始源石锭数量 = 起始状态.源石锭数量
            起始烛火数量 = 起始状态.烛火数量

            if 起始源石锭数量 < 50:
                每次尝试消耗源石锭数量 = 20
                最多可尝试次数 = 起始源石锭数量 // 每次尝试消耗源石锭数量
                尝试成功获得源石锭数量 = 30
                尝试成功获得烛火数量 = 3
            else:
                每次尝试消耗源石锭数量 = 50
                最多可尝试次数 = 起始源石锭数量 // 每次尝试消耗源石锭数量
                尝试成功获得源石锭数量 = 65
                尝试成功获得烛火数量 = 6

            if 起始烛火数量 <= 0 or 起始源石锭数量 < 20:  # 没有烛火或者源石锭，失败
                转移概率列表.append((起始状态, 1))

            elif 起始烛火数量 > 1:  # 有烛火，则拿源石锭
                for 尝试次数 in range(1, 最多可尝试次数 + 1):
                    概率 = self.投出至少1枚衡钱所需投钱次数的分布.pmf(尝试次数)
                    转移概率列表.append((self.获取状态(源石锭数量=起始源石锭数量 - 每次尝试消耗源石锭数量 * 尝试次数 + 尝试成功获得源石锭数量, 烛火数量=起始烛火数量 - 1), 概率))  # 若干次投钱后投出了1枚衡钱
                失败概率 = self.投出至少1枚衡钱所需投钱次数的分布.sf(最多可尝试次数)
                转移概率列表.append((self.获取状态(源石锭数量=起始源石锭数量 - 每次尝试消耗源石锭数量 * 最多可尝试次数, 烛火数量=起始烛火数量 - 1), 失败概率))  # 没有投出1枚衡钱

            else:  # 没有烛火，则拿烛火
                for 尝试次数 in range(1, 最多可尝试次数 + 1):
                    概率 = self.投出至少2枚厉钱所需投钱次数的分布.pmf(尝试次数)
                    转移概率列表.append((self.获取状态(源石锭数量=起始源石锭数量 - 每次尝试消耗源石锭数量 * 尝试次数, 烛火数量=起始烛火数量 - 1 + 尝试成功获得烛火数量), 概率))  # 若干次投钱后投出了2枚厉钱
                失败概率 = self.投出至少2枚厉钱所需投钱次数的分布.sf(最多可尝试次数)
                转移概率列表.append((self.获取状态(源石锭数量=起始源石锭数量 - 每次尝试消耗源石锭数量 * 最多可尝试次数, 烛火数量=起始烛火数量 - 1), 失败概率))  # 没有投出2枚厉钱

        assert np.isclose(sum(x[1] for x in 转移概率列表), 1)
        return 转移概率列表

    def 构建状态列表(self) -> None:
        self.状态列表: list[状态类] = []
        self.状态列表.extend(过渡态类(源石锭数量=源石锭数量, 烛火数量=烛火数量)
                         for 源石锭数量 in range(0, self.电表倒转成功源石锭阈值 + 1, 1) for 烛火数量 in range(0, 7, 1))
        self.状态列表.append(吸收态类.源石锭达到阈值)

        self.状态数量: int = len(self.状态列表)
        self.状态索引: dict[状态类, int] = {状态: i for i, 状态 in enumerate(self.状态列表)}

        self.失败状态序号列表: list[int] = [状态序号 for 状态序号, 状态 in enumerate(self.状态列表) if not isinstance(状态, 吸收态类) and (状态.源石锭数量 < 20 or 状态.烛火数量 <= 0)]
        self.成功状态序号列表: list[int] = [self.状态索引[吸收态类.源石锭达到阈值]]
        self.中间态序号列表: list[int] = [状态序号 for 状态序号, 状态 in enumerate(self.状态列表) if 状态序号 not in self.失败状态序号列表 and 状态序号 not in self.成功状态序号列表]

    def 构建状态转移矩阵(self) -> None:
        dok_状态转移矩阵 = scipy.sparse.dok_array((self.状态数量, self.状态数量))
        for 起始状态 in self.状态列表:
            起始状态索引 = self.状态索引[起始状态]
            转移概率列表 = self.状态转移(起始状态)
            for 目标状态, 概率 in 转移概率列表:
                目标状态索引 = self.状态索引[目标状态]
                dok_状态转移矩阵[起始状态索引, 目标状态索引] += 概率
        self.状态转移矩阵 = dok_状态转移矩阵.tocsr()


def 单个衡钱和厉钱数量计算(电表倒转成功源石锭阈值: int, 衡钱数量: int, 厉钱数量: int, 迭代次数: int, 初始源石锭数量范围: Iterable[int]) -> list[tuple[int, int, float, float, float]]:
    求解器 = 求解器类(电表倒转成功源石锭阈值=电表倒转成功源石锭阈值, 衡钱数量=衡钱数量, 厉钱数量=厉钱数量)
    求解器.构建状态列表()
    求解器.构建状态转移矩阵()
    print(f"状态转移矩阵构建完成，状态数量：{求解器.状态数量}")

    失败状态矩阵 = scipy.sparse.dok_array((len(失败状态序号列表), 状态数量))
    for i, 状态序号 in enumerate(失败状态序号列表):
        失败状态矩阵[i, 状态序号] = 1
    失败状态矩阵 = 失败状态矩阵.tocsr()

    成功状态矩阵 = scipy.sparse.dok_array((len(成功状态序号列表), 状态数量))
    for i, 状态序号 in enumerate(成功状态序号列表):
        成功状态矩阵[i, 状态序号] = 1
    成功状态矩阵 = 成功状态矩阵.tocsr()

    矩阵 = scipy.sparse.vstack([状态转移矩阵 - scipy.sparse.identity(状态数量, format='csr'), 失败状态矩阵, 成功状态矩阵])
    向量 = np.hstack([np.zeros(状态数量), np.zeros(len(失败状态序号列表)), np.ones(len(成功状态序号列表))])

    result = scipy.sparse.linalg.lsqr(矩阵, 向量, atol=0, btol=0)
    解 = result[0]
    print(result)

    for 源石锭数量 in range(0, 1025, 16):
        状态 = 过渡态类(源石锭数量=源石锭数量, 烛火数量=1)
        状态序号 = 状态索引[状态]
        成功概率 = 解[状态序号]
        print(f"源石锭数量：{源石锭数量:4}，成功概率：{成功概率:9.4%}")

    return []

    结果 = []
    for 初始源石锭数量 in 初始源石锭数量范围:
        初始状态 = 过渡态类(源石锭数量=初始源石锭数量, 烛火数量=1)
        当前状态分布 = np.zeros(状态数量)
        当前状态分布[状态索引[初始状态]] = 1
        for i in range(迭代次数):
            当前状态分布 = 当前状态分布 @ 状态转移矩阵

        失败概率 = np.sum(当前状态分布[失败状态序号列表])
        成功概率 = np.sum(当前状态分布[成功状态序号列表])
        中间态概率 = np.sum(当前状态分布[中间态序号列表])
        print(f"游客分队，衡钱数量：{衡钱数量}，厉钱数量：{厉钱数量}，初始源石锭数量：{初始源石锭数量:4}，失败概率：{失败概率:9.4%}，成功概率：{成功概率:9.4%}，中间态概率：{中间态概率:9.4%}")

        结果.append((衡钱数量, 厉钱数量, 初始源石锭数量, 失败概率, 成功概率, 中间态概率))
    return 结果


if __name__ == "__main__":
    电表倒转成功源石锭阈值 = 2048
    待计算的衡钱和厉钱数量范围 = [(4, 7), (4, 8), (5, 5), (5, 6), (5, 7), (6, 5), (6, 6), (7, 5)]
    迭代次数 = 8192
    初始源石锭数量范围 = range(0, 1024 + 1, 32)

    单个衡钱和厉钱数量计算(
        电表倒转成功源石锭阈值=电表倒转成功源石锭阈值,
        衡钱数量=5,
        厉钱数量=5,
        迭代次数=迭代次数,
        初始源石锭数量范围=初始源石锭数量范围
    )

    # with multiprocessing.Pool(processes=len(待计算的衡钱和厉钱数量范围)) as pool:
    #     results = pool.starmap(
    #         单个衡钱和厉钱数量计算,
    #         ((电表倒转成功源石锭阈值, 衡钱数量, 厉钱数量, 迭代次数, 初始源石锭数量范围) for 衡钱数量, 厉钱数量 in 待计算的衡钱和厉钱数量范围)
    #     )

    # 结果列表 = []
    # for 单个结果 in results:
    #     结果列表.extend(单个结果)

    # df = pd.DataFrame(结果列表, columns=["衡钱数量", "厉钱数量", "初始源石锭数量", "失败概率", "成功概率", "中间态概率"])
    # df.to_csv("游客分队源石锭电表倒转成功概率.csv", index=False)
