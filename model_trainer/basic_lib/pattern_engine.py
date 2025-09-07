



def has_adjacent_wildcards(pattern: str) -> bool:
    """
    返回 True 当且仅当 pattern 中存在相邻段，且至少一段是 '**' 且另一段是 '*' 或 '**'。
    禁止的相邻对：('**','**'), ('**','*'), ('*','**')
    允许的相邻对：('*','*')
    """
    if not pattern:
        return False
    segs = pattern.split('.')
    for a, b in zip(segs, segs[1:]):
        if (a == '**' and b in {'*', '**'}) or (a == '*' and b == '**'):
            return True
    return False


def filter_adjacent_wildcards(patterns: list[str], *, mode: str = "error") -> list[str]:
    """
    过滤包含“非法相邻通配符（涉及 '**'）”的模式。
    - mode = "error": 发现非法模式直接抛 ValueError
    - mode = "drop" : 丢弃非法模式，返回其余模式
    """
    invalid = [p for p in patterns if has_adjacent_wildcards(p)]
    if invalid:
        if mode == "error":
            raise ValueError(f"Adjacent wildcards near '**' not allowed: {invalid}")
        elif mode == "drop":
            return [p for p in patterns if p not in invalid]
        else:
            raise ValueError(f"Unknown mode={mode}, expected 'error' or 'drop'")
    return patterns


def has_empty_segments(pattern: str) -> bool:
    """
    返回 True 当且仅当 pattern 存在空段（结构不完整），例如：
    - 'a....b'（多重连点）
    - 'a.b.'（尾部有点）
    - '.a' 或 '.a.b'（头部有点）
    - 'a..*.b' 等

    实现：split('.') 后检查是否有空字符串段。
    """
    if pattern is None:
        return True
    # 不主动 strip，避免静默更改用户输入；只基于 '.' 拆分检查空段
    segs = pattern.split('.')
    return any(s == '' for s in segs)


def filter_empty_segment_patterns(patterns: list[str], *, mode: str = "error") -> list[str]:
    """
    过滤包含空段（结构不完整）的模式。
    - mode = "error": 发现非法直接抛 ValueError
    - mode = "drop" : 丢弃非法模式，返回其余模式
    """
    invalid = [p for p in patterns if has_empty_segments(p)]
    if invalid:
        if mode == "error":
            raise ValueError(f"Patterns with empty segments (e.g., a..b / a.b.): {invalid}")
        elif mode == "drop":
            return [p for p in patterns if p not in invalid]
        else:
            raise ValueError(f"Unknown mode={mode}, expected 'error' or 'drop'")
    return patterns


def filter_invalid_patterns(patterns: list[str], *, mode: str = "error") -> list[str]:
    """
    组合校验：
    - 相邻涉及 ** 的通配非法：**.*、*.**、**.**
    - 结构空段非法：a..b、a.b.、.a 等

    mode 同上：
    - 'error': 任一类非法命中则抛错并汇总
    - 'drop' : 丢弃所有非法，返回剩余
    """
    adj_invalid = [p for p in patterns if has_adjacent_wildcards(p)]
    empty_invalid = [p for p in patterns if has_empty_segments(p)]
    all_invalid = sorted(set(adj_invalid) | set(empty_invalid))

    if all_invalid:
        if mode == "error":
            msg_parts = []
            if adj_invalid:
                msg_parts.append(f"adjacent wildcards near '**': {sorted(set(adj_invalid))}")
            if empty_invalid:
                msg_parts.append(f"empty segments (e.g., a..b / a.b.): {sorted(set(empty_invalid))}")
            raise ValueError("Invalid patterns: " + " ; ".join(msg_parts))
        elif mode == "drop":
            return [p for p in patterns if p not in all_invalid]
        else:
            raise ValueError(f"Unknown mode={mode}, expected 'error' or 'drop'")
    return patterns


# from functools import lru_cache

def _pattern_stats(p: str) -> tuple[int, int, int, int, int]:
    """
    返回 (depth, star, dstar, literal, mixed):
    - depth: 总分段数
    - star:  '*' 段数量（按分段计数）
    - dstar: '**' 段数量
    - literal: 字面段数量 = depth - star - dstar
    - mixed: 是否混用（* 与 ** 同时出现）; 1=混用, 0=不混用
    """
    segs = p.split('.') if p else []
    depth = len(segs)
    star = sum(1 for s in segs if s == '*')
    dstar = sum(1 for s in segs if s == '**')
    literal = depth - star - dstar
    mixed = 1 if (star > 0 and dstar > 0) else 0
    return depth, star, dstar, literal, mixed


# @lru_cache(maxsize=None)
def _specificity_tuple(p: str) -> tuple[int, int, int, int, int]:
    """
    越“具体”的模式，返回的 tuple 越大。
    排序键（降序）：(literal, -dstar, -star, depth, -mixed)
    """
    depth, star, dstar, literal, mixed = _pattern_stats(p)
    return (literal, -dstar, -star, depth, -mixed)


def compare_patterns_depth_v2(pattern1: str, pattern2: str) -> int:
    """
    返回：-1 0 1
    -1 表示 p1 比 p2 更具体
     0 表示 两者在本比较下等价
     1 表示 p1 比 p2 更不具体
    """
    k1 = _specificity_tuple(pattern1)
    k2 = _specificity_tuple(pattern2)
    if k1 > k2:
        return -1
    elif k1 < k2:
        return 1
    else:
        return 0


def sort_patterns_v2(patterns: list[str]) -> list[str]:
    """
    按具体度从高到低排序；同具体度“后写优先”。
    使用稳定排序 + 原始索引作为末位分量，reverse=True 实现“后者覆盖前者”的直觉。
    """
    enriched = [(p, i) for i, p in enumerate(patterns)]
    enriched_sorted = sorted(
        enriched,
        key=lambda t: (_specificity_tuple(t[0]), t[1]),
        reverse=True,
    )
    return [p for p, _ in enriched_sorted]



def match_pattern(txt: str, pattern: str, *, allow_prefix: bool = False) -> bool:
    """
    段式通配匹配（迭代 DP，O(m·n) 时间，O(n) 空间）：
    - 从头匹配（start-anchored）
    - 默认精确到末尾（allow_prefix=False）；若 allow_prefix=True，则 pattern 耗尽即命中
    - '*'  匹配恰好一段
    - '**' 匹配零或多段
    - 非法模式（空段或涉及 '**' 的相邻通配）直接返回 False
    """

    p = pattern.split('.') if pattern else []
    k = txt.split('.') if txt else []
    m, n = len(p), len(k)

    # dp[i][j] 表示 p[i:] 能否匹配 k[j:]
    # 末行初始化：
    # - 精确匹配：仅 j==n 为 True
    # - 前缀匹配：整行 True（pattern 耗尽即命中）
    if allow_prefix:
        dp_next = [True] * (n + 1)
    else:
        dp_next = [False] * (n + 1)
        dp_next[n] = True

    for i in range(m - 1, -1, -1):
        token = p[i]
        dp_curr = [False] * (n + 1)

        if token == '**':
            # dp[i][j] = dp[i+1][j] or (j < n and dp[i][j+1])
            dp_curr[n] = dp_next[n]  # '**' 在 j==n 时只能吞 0 段
            for j in range(n - 1, -1, -1):
                dp_curr[j] = dp_next[j] or dp_curr[j + 1]

        elif token == '*':
            # dp[i][j] = (j < n and dp[i+1][j+1])
            dp_curr[n] = False
            for j in range(n - 1, -1, -1):
                dp_curr[j] = dp_next[j + 1]

        else:
            # 字面匹配：dp[i][j] = (j < n and p[i]==k[j] and dp[i+1][j+1])
            dp_curr[n] = False
            for j in range(n - 1, -1, -1):
                dp_curr[j] = (token == k[j]) and dp_next[j + 1]

        dp_next = dp_curr

    return dp_next[0]


def _normalize_core_pattern(p: str) -> str:
    # 排序时不让 '!' 影响具体度，仅匹配层面处理它
    return p[1:] if p and p[0] == '!' else p


def _pattern_stats_v3(p: str) -> tuple:
    """
    返回：
    - leading_literal: 开头连续字面段数量（遇到第一个通配符前的长度）
    - head_kind_score: 第一个通配符种类（2=无通配, 1='*', 0='**'）
    - literal, dstar, star, depth, mixed, star_only
    """
    core = _normalize_core_pattern(p) or ""
    segs = core.split('.') if core else []
    depth = len(segs)
    star = sum(1 for s in segs if s == '*')
    dstar = sum(1 for s in segs if s == '**')
    literal = depth - star - dstar
    mixed = 1 if (star > 0 and dstar > 0) else 0

    # leading literal run + 第一个通配符类型
    leading_literal = 0
    head_kind_score = 2  # 先假设“没有通配符”
    for s in segs:
        if s == '*':
            head_kind_score = 1
            break
        elif s == '**':
            head_kind_score = 0
            break
        else:
            leading_literal += 1

    star_only = (literal == 0 and dstar == 0 and star > 0)
    return leading_literal, head_kind_score, literal, dstar, star, depth, mixed, star_only


def _specificity_tuple_v3(p: str) -> tuple:
    """
    v3 具体度键（reverse=True 下越大越具体）：
    - 特判：纯星模式整体压到末尾，组内 depth 越大越具体（'*.*' > '*'）
    - 主序：leading_literal > head_kind_score > literal > (-dstar) > (-star) > depth > (-mixed)
    """
    leading_literal, head_kind_score, literal, dstar, star, depth, mixed, star_only = _pattern_stats_v3(p)
    if star_only:
        # 极小哨兵把纯星模式整体放到末尾；组内用 depth 排序（越深越具体）
        return (-10 ** 9, 0, 0, 0, 0, depth, 0)

    return (
        leading_literal,  # 越靠后才出现通配符越具体
        head_kind_score,  # 无通配 > '*' > '**'
        literal,  # 总字面越多越具体
        -dstar,  # '**' 越少越具体
        -star,  # '*' 越少越具体
        depth,  # 更深更具体（在上面因素相同时）
        -mixed  # 不混用更具体
    )


def sort_patterns_v3(patterns: list[str]) -> list[str]:
    enriched = [(p, i) for i, p in enumerate(patterns)]
    enriched_sorted = sorted(
        enriched,
        key=lambda t: (_specificity_tuple_v3(t[0]), t[1]),  # 同具体度“后写优先”
        reverse=True,
    )
    return [p for p, _ in enriched_sorted]


from functools import cmp_to_key


def compare_patterns_depth_v3(p1: str, p2: str) -> int:
    """
    v3 比较器（只按具体度比较，不含“后写优先”）。
    返回 -1/0/1：
      -1 表示 p1 比 p2 更具体
       0 表示 等价
       1 表示 p1 比 p2 更不具体
    具体度规则与 sort_patterns_v3 完全一致（_specificity_tuple_v3）。
    """
    k1 = _specificity_tuple_v3(p1)
    k2 = _specificity_tuple_v3(p2)
    if k1 > k2:
        return -1
    elif k1 < k2:
        return 1
    else:
        return 0


def compare_patterns_depth_v3_enriched(a: tuple[str, int], b: tuple[str, int]) -> int:
    """
    v3 比较器（带原始索引，内置“后写优先”tie-break）。
    a/b 形如: (pattern, index)；index 是原始出现顺序。
    返回 -1/0/1，排序升序即得到“更具体在前；同具体度后写在前”的结果。
    """
    p1, i1 = a
    p2, i2 = b
    k1 = _specificity_tuple_v3(p1)
    k2 = _specificity_tuple_v3(p2)

    # 先比具体度
    if k1 > k2:
        return -1
    if k1 < k2:
        return 1

    # 再比“后写优先”：较大的索引更优
    if i1 > i2:
        return -1
    if i1 < i2:
        return 1
    return 0


def sort_patterns_v3_cmp(patterns: list[str]) -> list[str]:
    """
    使用 v3 enriched 比较器的排序实现（可替代 sort_patterns_v3 的 key 版本）。
    结果语义一致：更具体在前；同具体度时，后写优先。
    """
    enriched = [(p, i) for i, p in enumerate(patterns)]
    enriched_sorted = sorted(enriched, key=cmp_to_key(compare_patterns_depth_v3_enriched))
    return [p for p, _ in enriched_sorted]






if __name__ == '__main__':

    def t1est_match_pattern():
        # 默认 allow_prefix=False（精确到末尾）
        cases = [
            # 精确匹配
            ("a.b", "a.b", False, True),
            ("a.b.c", "a.b", False, False),  # pattern 短，key 更长 → 不匹配
            ("a.b", "a.b.c", False, False),  # pattern 更长且无 ** → 不匹配

            # allow_prefix=True（前缀即命中）
            ("a.b.c", "a.b", True, True),  # 允许 key 有剩余

            # '*' 恰好一段
            ("a.b.c", "a.b.*", False, True),
            ("a.b.c.d", "a.b.*", False, False),  # * 只能吞一段

            # '**' 零或多段
            ("a.b.c.d", "a.b.**", False, True),
            ("a.b", "a.b.**", False, True),  # ** 吞 0 段
            ("a.b.c", "**", False, True),  # 任意

            # 组合
            ("a.b.c", "*.*", False, False),
            ("a.b.c", "*.*", True, True),
            ("a.b.c", "*.*.*", False, True),
            ("a.b.c", "*.b.*", False, True),
            ("a.b.c", "*.c.*", False, False),  # 中间 b != c

            # 你关心的：pattern 比 key 更长但无 '**' → 不匹配
            ("a.b", "a.b.b.b", False, False),
            ("a.b.b.b", "a.b", False, False),
        ]

        for txt, pat, allow_prefix, expected in cases:
            got = match_pattern(txt, pat, allow_prefix=allow_prefix)
            assert got == expected, f"match_pattern({txt!r}, {pat!r}, allow_prefix={allow_prefix}) -> {got}, expected {expected}"


    def t1est_invalid_patterns_filters():
        # 非法模式（相邻涉及 **）
        assert has_adjacent_wildcards("a.**.*") is True
        assert has_adjacent_wildcards("a.*.**") is True
        assert has_adjacent_wildcards("a.**.**") is True
        assert has_adjacent_wildcards("a.*.*") is False
        assert has_adjacent_wildcards("a.**.b") is False

        # 空段模式
        assert has_empty_segments("a..b") is True
        assert has_empty_segments("a.b.") is True
        assert has_empty_segments(".a.b") is True
        assert has_empty_segments("a.b") is False

        # 组合过滤（drop 模式）
        patterns = ["a..b", "a.b", "a.**.*", "x.y", "a.b.", "a.*.*"]
        dropped = filter_invalid_patterns(patterns, mode="drop")
        # 应仅保留合法的
        assert dropped == ["a.b", "x.y", "a.*.*"]


    def _is_star_only(p: str) -> bool:
        depth, star, dstar, literal, mixed = _pattern_stats(p)
        return literal == 0 and dstar == 0 and star > 0


    def t1est_sort_patterns_v2():
        demo = [
            "a.b.*.c",
            "a.**.b",
            "a.*.*.b",
            "a.**.b",  # 重复
            "*.*",
            "*",
            "encoder.**.layer.*",
            "a.b.c",
        ]
        res = sort_patterns_v2(demo)

        # 直觉 1：纯字面最具体在最前
        assert res[0] == "a.b.c"

        # 直觉 2：纯星模式整体靠后（具体顺序不强制，但应当在末尾区域）
        star_only_positions = [i for i, p in enumerate(res) if _is_star_only(p)]
        non_star_positions = [i for i, p in enumerate(res) if not _is_star_only(p)]
        assert len(star_only_positions) >= 1
        assert max(non_star_positions) < min(star_only_positions), \
            f"star-only patterns should be after all non-star patterns; got {res}"


    pass
    print(match_pattern("a.b.c", "*.*", allow_prefix=True))
    t1est_match_pattern()
    t1est_invalid_patterns_filters()
    t1est_sort_patterns_v2()
    # compare_patterns_depth('*', 'encoder.**.layer.*.a')
    #
    # pattern = 'a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p.q.r.s.t.u.v.w.x.y.z'
    # base_depth, extra_depth1, extra_depth2 = get_pattern_depth(pattern)
    # print(base_depth, extra_depth1, extra_depth2)
    # p=get_pattern_part_depth(pattern)
    # print(p)
    # pattern2 = 'a.b.c.d.e.**.g.h.i.*.k.l.m.n.**.*.q.**.s.t.u.v.w.x.y.z'
    # p=get_pattern_part_depth(pattern2)
    # print(p)
    # base_depth, extra_depth1, extra_depth2 = get_pattern_depth(pattern2)
    # print(base_depth, extra_depth1, extra_depth2)
    # # Demo for adjacent wildcard filtering
    # demo = ['a.b.*.c', 'a.**.b', 'a.*.*.b', 'a.**.*.b', '**.*', '*.*', 'encoder.**.layer.*','a.b.c.c.v.c.x.c']
    # print('has_adjacent:', {d: has_adjacent_wildcards(d) for d in demo})
    # print('filtered:', filter_adjacent_wildcards(demo, mode='drop'))
    demo = ['!a.b.*.c', 'a.**.b', 'a.*.*.b', 'a.**.b', '*.*', 'encoder.**.layer.*', 'a.b.c.c.v.c.x.c',
            'encoder.**.layer.*.a', 'a.b.*.b.c.b.*.*', '*', 'a.b.b.b.*', 'a.b.b.b.c.*.*.*', '*.*.*.*.','a.c.d.f.g.s.**','**.am.s.d.f.e.f.a.s.d','*.am.s.d.f.e.f.a.s.d']
    # print('sort:', sort_patterns(demo))
    print('sort_v2:', sort_patterns_v2(demo))
    print('sort_v3:', sort_patterns_v3(demo))
    print('sort_patterns_v3_cmp', sort_patterns_v3_cmp(demo))
    print(compare_patterns_depth_v3('a.b.b.b.*', 'a.b.*.b.c.b.*.*'))
    # print(compare_patterns_depth('*.*', 'a.b.*.c'))
    # print(compare_patterns_depth('*.*.*.*', 'a.b.*.c'))
    # print(compare_patterns_depth_v2('*.*.*.*.*', 'a.b.*.c.**'))
    # print(_specificity_tuple('*.a'))
    # print(_specificity_tuple('*.*'))
    # print(_specificity_tuple('a.a.*.**'))

    def get_pattern_depth(pattern: str) -> tuple[int, int, int]:
        # base_depth = pattern.count('.')
        # extra_depth1 = pattern.count('**')
        # extra_depth2 = pattern.count('*')
        segs = pattern.split('.')
        base_depth = len(segs)
        extra_depth1 = sum(1 for s in segs if s == '*')
        extra_depth2 = sum(1 for s in segs if s == '**')
        return base_depth, extra_depth1, extra_depth2


    def get_pattern_part_depth(pattern: str) -> list[list[int]]:
        ptc = pattern.split('.')
        prc = []
        s0 = 0
        for p in ptc:
            s0 += 1
            if p == '**':
                prc.append([s0, 1])
                s0 = 0
            elif p == '*':
                prc.append([s0, 0])
                s0 = 0
        if s0 > 0:
            prc.append([s0, -1])
        return prc

    def compare_patterns_depth(pattern1: str, pattern2: str) -> int:  # 不修了修不好了
        '''

        :param pattern1:
        :param pattern2:
        :return: -1 0 1 -1代表p1>p2 0代表p1==p2 1代表p1<p2
        '''
        base_depth1, extra_depth_pattern1_1, extra_depth_pattern1_2 = get_pattern_depth(pattern1)
        base_depth2, extra_depth_pattern2_1, extra_depth_pattern2_2 = get_pattern_depth(pattern2)

        if base_depth1 > base_depth2:
            if extra_depth_pattern1_1 == 0 and extra_depth_pattern1_2 == 0:
                return -1
            elif extra_depth_pattern1_1 > 0 and extra_depth_pattern1_2 == 0:
                if extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 == 0:
                    p_depth_pattern1 = get_pattern_part_depth(pattern1)
                    p_depth_pattern2 = get_pattern_part_depth(pattern2)
                    if len(p_depth_pattern1) == len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                        return 0
                    else:
                        if extra_depth_pattern1_1 > extra_depth_pattern2_1:
                            return 1
                        elif extra_depth_pattern1_1 < extra_depth_pattern2_1:
                            return -1
                        else:
                            return 0

                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 > 0:
                    return -1
                else:  # extra_depth_pattern2_1==0 and extra_depth_pattern2_2>0:
                    return -1

            elif extra_depth_pattern1_1 == 0 and extra_depth_pattern1_2 > 0:
                if extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 > 0:
                    return -1
                else:  # extra_depth_pattern2_1==0 and extra_depth_pattern2_2>0:
                    p_depth_pattern1 = get_pattern_part_depth(pattern1)
                    p_depth_pattern2 = get_pattern_part_depth(pattern2)
                    if len(p_depth_pattern1) == len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                        return 0
                    else:
                        if extra_depth_pattern1_2 > extra_depth_pattern2_2:
                            return 1
                        elif extra_depth_pattern1_2 < extra_depth_pattern2_2:
                            return -1
                        else:
                            return 0



            else:  # extra_depth_pattern1_1>0 and extra_depth_pattern1_2>0:
                if extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 > 0:
                    p_depth_pattern1 = get_pattern_part_depth(pattern1)
                    p_depth_pattern2 = get_pattern_part_depth(pattern2)
                    if len(p_depth_pattern1) == len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                            elif p_depth_pattern1[i][0] == p_depth_pattern2[i][0]:
                                if p_depth_pattern1[i][1] > p_depth_pattern2[i][1]:
                                    return -1
                                elif p_depth_pattern1[i][1] < p_depth_pattern2[i][1]:
                                    return 1
                        return 0
                    elif len(p_depth_pattern1) > len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern2)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                            elif p_depth_pattern1[i][0] == p_depth_pattern2[i][0]:
                                if p_depth_pattern1[i][1] > p_depth_pattern2[i][1]:
                                    return -1
                                elif p_depth_pattern1[i][1] < p_depth_pattern2[i][1]:
                                    return 1
                        return 0
                    else:  # len(p_depth_pattern1) < len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                            elif p_depth_pattern1[i][0] == p_depth_pattern2[i][0]:
                                if p_depth_pattern1[i][1] > p_depth_pattern2[i][1]:
                                    return -1
                                elif p_depth_pattern1[i][1] < p_depth_pattern2[i][1]:
                                    return 1
                        return 0

                else:  # extra_depth_pattern2_1==0 and extra_depth_pattern2_2>0:
                    return 1


        elif base_depth1 == base_depth2:
            if extra_depth_pattern1_1 == 0 and extra_depth_pattern1_2 == 0:
                if extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 == 0:
                    return 0
                else:
                    return -1
            elif extra_depth_pattern1_1 > 0 and extra_depth_pattern1_2 == 0:
                if extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 > 0:
                    return -1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 > 0:
                    return -1
                else:  # extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 == 0:
                    p_depth_pattern1 = get_pattern_part_depth(pattern1)
                    p_depth_pattern2 = get_pattern_part_depth(pattern2)
                    if len(p_depth_pattern1) == len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                        return 0
                    else:
                        if extra_depth_pattern1_1 > extra_depth_pattern2_1:
                            return 1
                        elif extra_depth_pattern1_1 < extra_depth_pattern2_1:
                            return -1
                        else:
                            return 0
            elif extra_depth_pattern1_1 > 0 and extra_depth_pattern1_2 > 0:
                if extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 > 0:
                    return 1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 > 0:
                    p_depth_pattern1 = get_pattern_part_depth(pattern1)
                    p_depth_pattern2 = get_pattern_part_depth(pattern2)
                    if len(p_depth_pattern1) == len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                            elif p_depth_pattern1[i][0] == p_depth_pattern2[i][0]:
                                if p_depth_pattern1[i][1] > p_depth_pattern2[i][1]:
                                    return -1
                                elif p_depth_pattern1[i][1] < p_depth_pattern2[i][1]:
                                    return 1
                        return 0
                    elif len(p_depth_pattern1) > len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern2)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                            elif p_depth_pattern1[i][0] == p_depth_pattern2[i][0]:
                                if p_depth_pattern1[i][1] > p_depth_pattern2[i][1]:
                                    return -1
                                elif p_depth_pattern1[i][1] < p_depth_pattern2[i][1]:
                                    return 1
                        return 0
                    else:  # len(p_depth_pattern1) < len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                            elif p_depth_pattern1[i][0] == p_depth_pattern2[i][0]:
                                if p_depth_pattern1[i][1] > p_depth_pattern2[i][1]:
                                    return -1
                                elif p_depth_pattern1[i][1] < p_depth_pattern2[i][1]:
                                    return 1
                        return 0
                else:  # extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 == 0:
                    return 1
            else:  # extra_depth_pattern1_1 == 0 and extra_depth_pattern1_2 > 0:
                if extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 > 0:
                    p_depth_pattern1 = get_pattern_part_depth(pattern1)
                    p_depth_pattern2 = get_pattern_part_depth(pattern2)
                    if len(p_depth_pattern1) == len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                        return 0
                    else:
                        if extra_depth_pattern1_2 > extra_depth_pattern2_2:
                            return 1
                        elif extra_depth_pattern1_2 < extra_depth_pattern2_2:
                            return -1
                        else:
                            return 0
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 > 0:
                    return -1
                else:  # extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 == 0:
                    return 1



        else:  # base_depth1 < base_depth2
            if extra_depth_pattern1_1 == 0 and extra_depth_pattern1_2 == 0:
                if extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 == 0:
                    return 1
                else:
                    return -1
            elif extra_depth_pattern1_1 > 0 and extra_depth_pattern1_2 == 0:
                if extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 == 0:
                    p_depth_pattern1 = get_pattern_part_depth(pattern1)
                    p_depth_pattern2 = get_pattern_part_depth(pattern2)
                    if len(p_depth_pattern1) == len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return 1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return -1
                        return 0  # 不可能走到这里
                    else:
                        if extra_depth_pattern1_1 > extra_depth_pattern2_1:
                            return 1
                        elif extra_depth_pattern1_1 < extra_depth_pattern2_1:
                            return -1
                        else:
                            return 0  # 不可能走到这里
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 > 0:
                    return -1
                else:  # extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 > 0:
                    return -1

            elif extra_depth_pattern1_1 == 0 and extra_depth_pattern1_2 > 0:
                if extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 > 0:
                    return -1
                else:  # extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 > 0:
                    p_depth_pattern1 = get_pattern_part_depth(pattern1)
                    p_depth_pattern2 = get_pattern_part_depth(pattern2)
                    if len(p_depth_pattern1) == len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return 1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return -1
                        return 0
                    else:
                        if extra_depth_pattern1_2 > extra_depth_pattern2_2:
                            return 1
                        elif extra_depth_pattern1_2 < extra_depth_pattern2_2:
                            return -1
                        else:
                            return 0

            else:  # extra_depth_pattern1_1>0 and extra_depth_pattern1_2>0:
                if extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 == 0:
                    return 1
                elif extra_depth_pattern2_1 > 0 and extra_depth_pattern2_2 > 0:
                    p_depth_pattern1 = get_pattern_part_depth(pattern1)
                    p_depth_pattern2 = get_pattern_part_depth(pattern2)
                    if len(p_depth_pattern1) == len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                            elif p_depth_pattern1[i][0] == p_depth_pattern2[i][0]:
                                if p_depth_pattern1[i][1] > p_depth_pattern2[i][1]:
                                    return 1
                                elif p_depth_pattern1[i][1] < p_depth_pattern2[i][1]:
                                    return -1
                        return 0
                    elif len(p_depth_pattern1) > len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern2)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                            elif p_depth_pattern1[i][0] == p_depth_pattern2[i][0]:
                                if p_depth_pattern1[i][1] > p_depth_pattern2[i][1]:
                                    return 1
                                elif p_depth_pattern1[i][1] < p_depth_pattern2[i][1]:
                                    return -1
                        return 0
                    else:  # len(p_depth_pattern1) < len(p_depth_pattern2):
                        for i in range(len(p_depth_pattern1)):
                            if p_depth_pattern1[i][0] > p_depth_pattern2[i][0]:
                                return -1
                            elif p_depth_pattern1[i][0] < p_depth_pattern2[i][0]:
                                return 1
                            elif p_depth_pattern1[i][0] == p_depth_pattern2[i][0]:
                                if p_depth_pattern1[i][1] > p_depth_pattern2[i][1]:
                                    return 1
                                elif p_depth_pattern1[i][1] < p_depth_pattern2[i][1]:
                                    return -1
                        return 0
                else:  # extra_depth_pattern2_1 == 0 and extra_depth_pattern2_2 > 0:
                    return 1


    def sort_patterns(patterns: list[str]) -> list[str]:
        # patterns_out=[None]*len(patterns)
        patterns_out = []
        patterns_with_index = [(i, p) for i, p in enumerate(patterns)]
        patterns_with_index = reversed(patterns_with_index)
        for i in patterns_with_index:
            if len(patterns_out) == 0:
                patterns_out.append(i)
            else:
                po_size = len(patterns_out)
                for j in range(po_size):
                    if compare_patterns_depth(i[1], patterns_out[j][1]) == 1:
                        if j == po_size - 1:
                            patterns_out.append(i)
                            break
                        continue
                    elif compare_patterns_depth(i[1], patterns_out[j][1]) == 0:
                        if i[0] <= patterns_out[j][0]:
                            if j == po_size - 1:
                                patterns_out.append(i)
                                break
                            continue
                        else:
                            patterns_out.insert(j, i)
                            break
                    else:  # compare_patterns_depth(i[1],patterns_out[j][1])==-1
                        patterns_out.insert(j, i)
                        break

        pox = [i[1] for i in patterns_out]
        return pox


