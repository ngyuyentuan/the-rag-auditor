from src.utils.normalize import normalize_text

def test_none_and_empty():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""

def test_only_punct():
    assert normalize_text("!!!???") == ""

def test_basic_lower_and_spaces():
    assert normalize_text("  Hello   World ") == "hello world"

def test_nfkc_equivalence():
    assert normalize_text("ＡＢＣ１２３") == "abc123"

def test_unicode_line_paragraph_separators():
    s = "a\u2028b\u2029c"
    assert normalize_text(s) == "a b c"

def test_keep_slash_dash_true_vs_false():
    s = "Form-10-K/2023"
    assert normalize_text(s, keep_slash_dash=False) == "form 10 k 2023"
    assert normalize_text(s, keep_slash_dash=True) == "form-10-k/2023"

def test_keep_dot_decimal():
    s = "Rate is 3.14%!"
    assert normalize_text(s, keep_dot=False) == "rate is 3 14"
    assert normalize_text(s, keep_dot=True) == "rate is 3.14"

def test_cjk():
    assert normalize_text("你好 世界") == "你好 世界"

def test_emoji_removed():
    assert normalize_text("Hello 😊") == "hello"

def test_dots_removed_by_default():
    assert normalize_text("...Hello...") == "hello"
