VIETNAMESE_REFUTES_PATTERNS = [
    r'\bkhông\s+(?:phải|có|thể)\b',
    r'\bchưa\s+(?:bao\s+giờ|từng)\b',
    r'\bkhông\s+bao\s+giờ\b',
    r'\bchỉ\s+(?:có|là)\b',
    r'\bduy\s+nhất\b',
    r'\bsai\b',
    r'\bkhông\s+đúng\b',
    r'\bkhông\s+chính\s+xác\b',
]

VIETNAMESE_SUPPORTS_PHRASES = [
    'là', 'là một', 'được', 'đã',
    'sinh ra', 'thành lập', 'phát hành',
    'đóng vai', 'xuất hiện', 'đạo diễn',
]

VIETNAMESE_NEI_PATTERNS = [
    r'\bcó\s+thể\b',
    r'\bcó\s+lẽ\b',
    r'\bchưa\s+rõ\b',
    r'\bchưa\s+xác\s+định\b',
    r'\bkhông\s+chắc\b',
]

NATIONALITY_VI = {
    'mỹ': 'usa', 'american': 'usa',
    'anh': 'uk', 'british': 'uk', 'english': 'uk',
    'pháp': 'france', 'french': 'france',
    'đức': 'germany', 'german': 'germany',
    'nhật': 'japan', 'japanese': 'japan',
    'hàn quốc': 'korea', 'korean': 'korea',
    'việt nam': 'vietnam', 'vietnamese': 'vietnam',
    'trung quốc': 'china', 'chinese': 'china',
}
