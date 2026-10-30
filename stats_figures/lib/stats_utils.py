import numpy as np
import pandas as pd


def p_to_asterisks(p, max_stars=4):
    """p値を有意性マーク（* ～ max_stars個の*, n.s.）に変換する"""
    if pd.isna(p):
        return '-'
    thresholds = [0.05, 0.01, 0.001, 0.0001, 0.00001][:max_stars]
    stars = sum(p < t for t in thresholds)
    return '*' * stars if stars else 'n.s.'


def format_p_scientific(p, max_stars=4):
    """P値を '6.27e-8 ****' のような文字列にする（表示用）"""
    if pd.isna(p):
        return '-'
    sig = p_to_asterisks(p, max_stars)
    if p <= 0 or np.isinf(p):
        return f'< 1e-15 {sig}'
    if p < 0.0001:
        exponent = int(np.floor(np.log10(p)))
        mantissa = p / (10 ** exponent)
        return f'{mantissa:.2f}e{exponent} {sig}'
    return f'{p:.4f} {sig}'
