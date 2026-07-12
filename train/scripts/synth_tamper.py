"""
Synthetic text field generation and ink tamper effects.

Extracted from the live notebook cell 58 (not the stale artifact).
Only change: imports forensic_panel_from_crop from local module.
"""
from __future__ import annotations
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from .forensic_panels import forensic_panel_from_crop

SEED = 42

SYNTH_FIRST_NAMES = [
    'MARIE', 'JEAN', 'AHMED', 'FATIMA', 'MOHAMED', 'AISHA', 'JOSE', 'ANA', 'PIERRE',
    'MARTHE', 'KOFFI', 'AMINATA', 'SOULEYMANE', 'IBRAHIMA', 'RACHID', 'NADIA',
    'ELENA', 'DAVID', 'SARAH', 'YAO', 'MOUSSA', 'GRACE', 'RAVI', 'PRIYA',
]
SYNTH_LAST_NAMES = [
    'DIALLO', 'TRAORE', 'KONE', 'CAMARA', 'SANTOS', 'PEREIRA', 'ALI', 'HASSAN',
    'IBRAHIM', 'MARTINS', 'DOSSO', 'MENSAH', 'RAMGULAM', 'ANOUK', 'SMITH',
    'JOSEPH', 'EL SAYED', 'NASSER', 'COULIBALY', 'BARRY', 'FERNANDES',
]
SYNTH_CITIES = [
    'CAIRO', 'ALEXANDRIA', 'PORT LOUIS', 'BEAU BASSIN', 'MAPUTO', 'BEIRA',
    'CONAKRY', 'KANKAN', 'COTONOU', 'PORTO-NOVO', 'PARAKOU', 'NAMPULA',
    'GIZA', 'MAHEBOURG', 'BOKE', 'NATITINGOU', 'PIERREBOURG', 'ROSE HILL',
]
SYNTH_STREETS = [
    '12 RUE VICTORIA PORT LOUIS',
    '45 AVENUE DE LA REPUBLIQUE COTONOU',
    '8 LOTISSEMENT BEL AIR ROSE HILL',
    '1027 ROUTE NATIONALE CONAKRY',
    'BLOC C RESIDENCE EL NOUR CAIRO',
    'RUA DA INDEPENDENCIA 214 MAPUTO',
    'QUARTIER ZONGO BP 184 PARAKOU',
]
SYNTH_AUTHORITIES = [
    'MINISTRY OF TRANSPORT AND ROAD SAFETY',
    'DIRECTION GENERALE DES TRANSPORTS TERRESTRES',
    'NATIONAL IDENTITY MANAGEMENT OFFICE',
    'TRAFFIC REGISTRATION AND LICENSING AUTHORITY',
    'SERVICE DES TITRES SECURISES',
]

SYNTH_FIELD_LABELS = {
    'name': ['NAME', 'NOM', 'FULL NAME', 'SURNAME'],
    'dob': ['DATE OF BIRTH', 'DOB', 'DATE NAISSANCE', 'NE(E) LE'],
    'issue_date': ['ISSUE DATE', 'DATE DELIVRANCE', 'DELIVRE LE'],
    'expiry_date': ['EXPIRY DATE', 'VALID UNTIL', 'EXPIRE LE'],
    'id_number': ['ID NO', 'N ID', 'NUMERO', 'PERMIS NO'],
    'city': ['PLACE', 'LIEU', 'VILLE'],
    'address': ['ADDRESS', 'ADRESSE', 'RESIDENCE'],
    'authority': ['ISSUING AUTHORITY', 'AUTORITE', 'DELIVRE PAR'],
    'category': ['CATEGORY', 'CATEGORIE', 'CLASS'],
}
SYNTH_FIELD_LABELS['name'] += ['NAME', 'VORNAME', 'اسم', 'نام']
SYNTH_FIELD_LABELS['dob'] += ['GEBURTSDATUM', 'تاريخ الميلاد', 'تاریخ پیدائش']
SYNTH_FIELD_LABELS['issue_date'] += ['AUSSTELLUNG', 'تاريخ الإصدار']
SYNTH_FIELD_LABELS['expiry_date'] += ['ABLAUFDATUM', 'gültig bis', 'تاريخ الانتهاء']
SYNTH_FIELD_LABELS['address'] += ['ANSCHRIFT', 'العنوان', 'پتہ']
SYNTH_FIELD_LABELS['city'] += ['STADT', 'ORT', 'مدينة']

SYNTH_MULTILINGUAL_NAMES = [
    'JÜRGEN MÜLLER', 'ANNA SCHÄFER', 'MARTIN GROẞ', 'FRANÇOIS KÉÏTA',
    'AMADOU N’DIAYE', 'SÉKOU TOURÉ', 'ỌLÁJÍDÉ ADEYEMI', 'CHINWE OKAFOR',
    'محمد علی', 'فاطمہ خان', 'أحمد حسن', 'عائشہ بی بی',
]
SYNTH_MULTILINGUAL_CITIES = [
    'MÜNCHEN', 'KÖLN', 'ZÜRICH', 'DAKAR', 'ABIDJAN', 'OUAGADOUGOU',
    'نواكشوط', 'القاهرة', 'کراچی', 'لاہور',
]
# Private test has unknown countries/scripts, so use the whole installed font library
# (which already covers Latin, Cyrillic, Greek, Arabic, Hebrew, CJK, Devanagari, Thai,
# Armenian, Georgian, Myanmar, etc.) plus locally-downloaded faces like OCR-B.
# choose_synth_font() still filters per-text by glyph coverage, so unusable fonts are skipped.
SYNTH_FONT_DIRS = [
    Path('fonts'),                                   # locally downloaded (OCR-B, etc.)
    Path('/System/Library/Fonts'),
    Path('/System/Library/Fonts/Supplemental'),
    Path('/Library/Fonts'),
    Path.home() / 'Library' / 'Fonts',
]
# Families that never appear as printed ID field text (emoji/symbol/dingbat/handwriting/extreme display).
SYNTH_FONT_BLOCKLIST = (
    # symbol / emoji / dingbat / braille
    'emoji', 'symbol', 'wingding', 'webding', 'dingbat', 'ornament', 'applebraille', 'braille',
    'lastresort', 'notocoloremoji', 'applecoloremoji',
    # Latin handwriting / script / extreme display (never printed on ID fields)
    'zapfino', 'chalkduster', 'comicsans', 'partylet', 'jazzlet', 'brushscript', 'snellroundhand',
    'savoye', 'applechancery', 'markerfelt', 'noteworthy', 'bradleyhand', 'trattatello',
    'herculanum', 'papyrus', 'sandiego', 'desdemona', 'playbill', 'stencil', 'luminari', 'phosphate',
    # calligraphic CJK brush styles (standard CJK still covered by PingFang/Hiragino/Songti/STSong)
    'hannotate', 'hanzipen', 'wawati', 'weibei', 'xingkai', 'yuppy', 'baoli', 'libian',
    # calligraphic Arabic (standard Arabic still covered by Geeza/Damascus/Naskh/SFArabic/Arial)
    'diwan', 'nastaleeq', 'farisi', 'mishafi', 'thuluth', 'kufistandard', 'corsiva',
)

def _font_is_blocked(name):
    key = name.lower().replace(' ', '').replace('-', '').replace('_', '')
    return any(b in key for b in SYNTH_FONT_BLOCKLIST)

def discover_synth_fonts():
    seen, paths = set(), []
    for d in SYNTH_FONT_DIRS:
        if not d.exists():
            continue
        for ext in ('*.ttf', '*.ttc', '*.otf'):
            for p in sorted(d.glob(ext)):
                if p.name in seen or _font_is_blocked(p.name):
                    continue
                seen.add(p.name)
                paths.append(p)
    return paths

SYNTH_FONT_PATHS = discover_synth_fonts()

def _font_class(name):
    n = name.lower().replace(' ', '')
    if any(k in n for k in ('ocr-b', 'ocrb', 'mono', 'courier', 'menlo', 'consol', 'andale', 'lettergothic')):
        return 'mono'
    if any(k in n for k in ('condensed', 'narrow')):
        return 'condensed'
    return 'normal'

def _font_is_italic(name):
    n = name.lower()
    return 'italic' in n or 'oblique' in n

_SYNTH_FONT_CHARMAP_CACHE = {}

def font_codepoints(path):
    key = str(path)
    if key not in _SYNTH_FONT_CHARMAP_CACHE:
        cmap = set()
        try:
            from fontTools.ttLib import TTFont
            # PIL renders index 0 of a .ttc collection, so read codepoints from face 0 too.
            font = TTFont(key, lazy=True, fontNumber=0) if key.lower().endswith('.ttc') else TTFont(key, lazy=True)
            for table in font['cmap'].tables:
                if table.isUnicode():
                    cmap.update(table.cmap.keys())
            font.close()
        except Exception:
            cmap = set()
        _SYNTH_FONT_CHARMAP_CACHE[key] = cmap
    return _SYNTH_FONT_CHARMAP_CACHE[key]

def text_codepoints(text):
    return sorted({ord(ch) for ch in str(text) if not ch.isspace()})

def font_support_fraction(path, text):
    cps = text_codepoints(text)
    if not cps:
        return 1.0
    cmap = font_codepoints(path)
    if not cmap:
        return 0.0
    return sum(cp in cmap for cp in cps) / len(cps)

def choose_synth_font(rng, size, text='', prefer=None, italic_p=0.0):
    """Pick a font that fully covers `text`.

    prefer: 'mono' / 'condensed' biases toward that class when available (e.g. mono for
            ID numbers / MRZ-style fields, condensed for long address/authority fields).
    italic_p: probability of preferring an italic/oblique face (ICAO uses italics for the
            second language); falls back to upright if none cover the text.
    """
    paths = [p for p in SYNTH_FONT_PATHS if p.exists()]
    if not paths:
        return ImageFont.load_default()
    cps = text_codepoints(text)
    if cps:
        covering = [p for p in paths if font_support_fraction(p, text) >= 1.0]
        if not covering:
            best = max(paths, key=lambda p: font_support_fraction(p, text))
            return ImageFont.truetype(str(best), size=size)
    else:
        covering = paths
    pool = covering
    # Class bias: prefer the requested class ~80% of the time when it can render the text.
    if prefer in ('mono', 'condensed'):
        classed = [p for p in covering if _font_class(p.name) == prefer]
        if classed and rng.random() < 0.80:
            pool = classed
    # Italic bias for second-language style captions.
    if italic_p > 0 and rng.random() < italic_p:
        ital = [p for p in pool if _font_is_italic(p.name)]
        if ital:
            pool = ital
    return ImageFont.truetype(str(pool[int(rng.integers(0, len(pool)))]), size=size)

def font_support_report(texts):
    rows = []
    for path in [p for p in SYNTH_FONT_PATHS if p.exists()]:
        for text in texts:
            cps = text_codepoints(text)
            cmap = font_codepoints(path)
            missing = ''.join(chr(cp) for cp in cps if cp not in cmap)
            rows.append({
                'font': path.name,
                'text': str(text),
                'support': round(font_support_fraction(path, text), 3),
                'missing_chars': missing,
            })
    return pd.DataFrame(rows).sort_values(['text', 'support', 'font'], ascending=[True, False, True]).reset_index(drop=True)

def pil_text_size(text, font):
    bbox = ImageDraw.Draw(Image.new('RGB', (8, 8))).textbbox((0, 0), str(text), font=font)
    return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])

def pil_put_text(img, text, pos, font, color, blur=False):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.text(pos, str(text), fill=(int(color), int(color), int(color)), font=font)
    out = np.array(pil)
    if blur:
        out = cv2.GaussianBlur(out, (3, 3), 0.45)
    img[:] = out

USE_EXTERNAL_SYNTH_CORPORA = True
SYNTH_CORPUS_CACHE = Path('artifacts') / 'synthetic_text_corpora'
SYNTH_CORPUS_URLS = {
    'forenames': 'https://raw.githubusercontent.com/sigpwned/popular-names-by-country-dataset/main/common-forenames-by-country.csv',
    'surnames': 'https://raw.githubusercontent.com/sigpwned/popular-names-by-country-dataset/main/common-surnames-by-country.csv',
    'cities': 'https://raw.githubusercontent.com/datasets/world-cities/main/data/world-cities.csv',
}

def read_cached_csv(url, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_csv(cache_path)
    df = pd.read_csv(url)
    df.to_csv(cache_path, index=False)
    return df

def unique_sample(series, n, random_state=SEED):
    values = series.dropna().astype(str).str.strip()
    values = values[values.ne('')].drop_duplicates()
    if len(values) == 0:
        return []
    return values.sample(min(n, len(values)), random_state=random_state).tolist()

def load_external_synth_corpora():
    corpora = {'forenames': [], 'surnames': [], 'cities': []}
    if not USE_EXTERNAL_SYNTH_CORPORA:
        return corpora
    try:
        forenames = read_cached_csv(SYNTH_CORPUS_URLS['forenames'], SYNTH_CORPUS_CACHE / 'common-forenames-by-country.csv')
        surnames = read_cached_csv(SYNTH_CORPUS_URLS['surnames'], SYNTH_CORPUS_CACHE / 'common-surnames-by-country.csv')
        cities = read_cached_csv(SYNTH_CORPUS_URLS['cities'], SYNTH_CORPUS_CACHE / 'world-cities.csv')
        preferred_countries = {'DE', 'FR', 'NG', 'ZA', 'EG', 'PK', 'IN', 'MA', 'SN', 'CI', 'GH', 'KE'}
        f = forenames[forenames['Country'].isin(preferred_countries)].copy()
        s = surnames[surnames['Country'].isin(preferred_countries)].copy()
        if len(f) == 0:
            f = forenames
        if len(s) == 0:
            s = surnames
        corpora['forenames'] = unique_sample(pd.concat([f['Localized Name'], f['Romanized Name']]), 800)
        corpora['surnames'] = unique_sample(pd.concat([s['Localized Name'], s['Romanized Name']]), 800)
        corpora['cities'] = unique_sample(cities['name'], 1200)
        print('Loaded external synthetic corpora:', {k: len(v) for k, v in corpora.items()})
    except Exception as e:
        print('External synthetic corpora unavailable; using embedded lists only.')
        print(type(e).__name__, str(e)[:300])
    return corpora

SYNTH_EXTERNAL_CORPORA = load_external_synth_corpora()
SYNTH_FIRST_NAMES = sorted(set(SYNTH_FIRST_NAMES + SYNTH_EXTERNAL_CORPORA['forenames']))
SYNTH_LAST_NAMES = sorted(set(SYNTH_LAST_NAMES + SYNTH_EXTERNAL_CORPORA['surnames']))
SYNTH_CITIES = sorted(set(SYNTH_CITIES + SYNTH_EXTERNAL_CORPORA['cities'] + SYNTH_MULTILINGUAL_CITIES))
SYNTH_TAMPER_TYPES = ['smooth_row_shift', 'copy_move', 'non_ink_background_noise', 'faint_horizontal_lines', 'field_static_noise', 'clean_text_reprint', 'cool_blue_bg_filter', 'diffusion_spread', 'bg_color_morph']
SYNTH_FIELD_STYLES = ['value_only', 'label_value_colon', 'label_value_space', 'boxed_label_value']

# Eastern-Arabic (Egypt) and Persian/Urdu numeral variants. Real Arabic-script IDs print
# dates/numbers in these glyphs; choose_synth_font() then auto-selects an Arabic-capable face.
_EASTERN_ARABIC_DIGITS = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')
_PERSIAN_DIGITS = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')

def maybe_localize_digits(text, rng):
    r = rng.random()
    if r < 0.12:
        return str(text).translate(_EASTERN_ARABIC_DIGITS)   # Egypt / Arabic
    if r < 0.18:
        return str(text).translate(_PERSIAN_DIGITS)          # Urdu / Persian
    return str(text)

def synth_random_date(rng):
    year = int(rng.integers(1935, 2032))
    month = int(rng.integers(1, 13))
    day = int(rng.integers(1, 29))
    formats = [
        f'{day:02d}/{month:02d}/{year}',
        f'{day:02d}-{month:02d}-{year}',
        f'{year}-{month:02d}-{day:02d}',
        f'{day:02d}.{month:02d}.{year}',
    ]
    return maybe_localize_digits(rng.choice(formats), rng)

def synth_random_id(rng):
    patterns = [
        ''.join(rng.choice(list('0123456789'), size=int(rng.integers(8, 13)))),
        '-'.join(''.join(rng.choice(list('0123456789'), size=n)) for n in [3, 4, 4]),
        ''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), size=int(rng.integers(7, 11)))),
        f'{rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))}{int(rng.integers(1000000, 9999999))}',
    ]
    return maybe_localize_digits(rng.choice(patterns), rng)

def synth_random_name(rng, long=False):
    if rng.random() < 0.22:
        return rng.choice(SYNTH_MULTILINGUAL_NAMES)
    n_parts = int(rng.integers(3, 6)) if long else (3 if rng.random() < 0.35 else 2)
    parts = []
    for i in range(n_parts):
        pool = SYNTH_FIRST_NAMES if i == 0 or (i < n_parts - 1 and rng.random() < 0.45) else SYNTH_LAST_NAMES
        parts.append(rng.choice(pool))
    return ' '.join(parts)

def synth_random_value(kind, rng, long=False):
    if kind in ['dob', 'issue_date', 'expiry_date']:
        return synth_random_date(rng)
    if kind == 'name':
        return synth_random_name(rng, long=long)
    if kind == 'city':
        return rng.choice(SYNTH_MULTILINGUAL_CITIES if rng.random() < 0.30 else SYNTH_CITIES)
    if kind == 'address':
        if rng.random() < 0.20:
            return rng.choice(['MÜLLERSTRAẞE 14 KÖLN', 'شارع النيل ٢٤ القاهرة', 'گلبرگ روڈ ۷ لاہور'])
        return rng.choice(SYNTH_STREETS) if long or rng.random() < 0.70 else rng.choice(SYNTH_CITIES)
    if kind == 'authority':
        if rng.random() < 0.20:
            return rng.choice(['BUNDESDRUCKEREI BERLIN', 'وزارة الداخلية', 'محکمہ ٹرانسپورٹ'])
        return rng.choice(SYNTH_AUTHORITIES) if long or rng.random() < 0.80 else rng.choice(['PREFECTURE', 'TRANSPORT OFFICE', 'POLICE HQ'])
    if kind == 'category':
        return rng.choice(['A', 'B', 'C', 'D', 'A,B', 'B,C,D', 'AM', 'BE'])
    return synth_random_id(rng)

def synth_splice_value(clean_value, fake_value, rng):
    clean_parts = str(clean_value).split()
    fake_parts = str(fake_value).split()
    if len(clean_parts) >= 3 and len(fake_parts) >= 1:
        start_word = int(rng.integers(1, len(clean_parts)))
        end_word = min(len(clean_parts), start_word + int(rng.integers(1, 3)))
        repl = fake_parts[:max(1, min(len(fake_parts), end_word - start_word))]
        out_parts = clean_parts[:start_word] + repl + clean_parts[end_word:]
        prefix = ' '.join(out_parts[:start_word])
        fake_text = ' '.join(repl)
        start = len(prefix) + (1 if prefix else 0)
        return ' '.join(out_parts), (start, start + len(fake_text))
    if any(sep in str(clean_value) for sep in ['/', '-', '.']):
        sep = next((s for s in ['/', '-', '.'] if s in str(clean_value)), '/')
        cp = str(clean_value).split(sep)
        fp = str(fake_value).split(sep)
        if len(cp) == len(fp) and len(cp) >= 2:
            idx = int(rng.integers(0, len(cp)))
            cp[idx] = fp[idx]
            out = sep.join(cp)
            start = sum(len(p) for p in cp[:idx]) + idx * len(sep)
            return out, (start, start + len(cp[idx]))
    n = len(str(clean_value))
    span = max(2, min(n, int(rng.integers(2, max(3, min(7, n + 1))))))
    start = int(rng.integers(0, max(1, n - span + 1)))
    repl = str(fake_value)[:span].ljust(span, 'X')
    out = str(clean_value)[:start] + repl + str(clean_value)[start + span:]
    return out, (start, start + span)

def synth_text_spec(seed, label, force_tamper_type=None):
    rng = np.random.default_rng(seed)
    field_kind = rng.choice(list(SYNTH_FIELD_LABELS.keys()), p=[0.18, 0.12, 0.11, 0.11, 0.16, 0.08, 0.10, 0.08, 0.06])
    field_style = rng.choice(SYNTH_FIELD_STYLES, p=[0.42, 0.30, 0.20, 0.08])
    is_long_value = bool(field_kind in ['name', 'address', 'authority'] and rng.random() < 0.45)
    label_text = rng.choice(SYNTH_FIELD_LABELS[field_kind])
    clean_value = synth_random_value(field_kind, rng, long=is_long_value)
    fake_value = synth_random_value(field_kind, rng, long=is_long_value)
    while fake_value == clean_value:
        fake_value = synth_random_value(field_kind, rng, long=is_long_value)
    if field_style == 'value_only':
        target_span = rng.choice(['value', 'value_part'], p=[0.55, 0.45])
    else:
        target_span = rng.choice(['value', 'value_part', 'label', 'both', 'background'], p=[0.40, 0.25, 0.08, 0.12, 0.15])
    tamper_type = 'clean' if label == 0 else (force_tamper_type or rng.choice(SYNTH_TAMPER_TYPES))
    fake_span_chars = None
    if label and target_span == 'value_part':
        final_value, fake_span_chars = synth_splice_value(clean_value, fake_value, rng)
    else:
        final_value = fake_value if label and target_span in ['value', 'both'] else clean_value
    final_label = label_text + rng.choice(['', '.', ' :']) if label and target_span in ['label', 'both'] else label_text
    value_segments = [('normal', final_value)]
    if fake_span_chars is not None:
        a, b = fake_span_chars
        value_segments = []
        if a > 0:
            value_segments.append(('normal', final_value[:a]))
        value_segments.append(('fake_part', final_value[a:b]))
        if b < len(final_value):
            value_segments.append(('normal', final_value[b:]))
    return {
        'field_kind': field_kind,
        'field_style': field_style,
        'is_long_value': is_long_value,
        'label_text': label_text,
        'value_text': clean_value,
        'fake_value_text': fake_value if label else '',
        'final_label_text': final_label,
        'final_value_text': final_value,
        'value_segments': value_segments,
        'fake_span_chars': fake_span_chars,
        'target_span': target_span if label else 'none',
        'tamper_type': tamper_type,
    }

SYNTH_BACKGROUND_MODES = ['rosette', 'rings', 'checker', 'woven', 'contour', 'microgrid', 'moire', 'dots']

def random_math_texture(h, w, rng, mode=None):
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx / max(w - 1, 1) - 0.5) * 2
    y = (yy / max(h - 1, 1) - 0.5) * 2
    cx = rng.uniform(-0.35, 0.35)
    cy = rng.uniform(-0.35, 0.35)
    xr = x - cx
    yr = y - cy
    r = np.sqrt(xr * xr + yr * yr)
    theta = np.arctan2(yr, xr)
    mode = mode or rng.choice(SYNTH_BACKGROUND_MODES)

    if mode == 'rosette':
        k1, k2 = rng.uniform(14, 28), rng.uniform(8, 20)
        m1, m2 = rng.uniform(3, 9), rng.uniform(5, 13)
        texture = np.sin(k1 * r + m1 * theta) + 0.7 * np.sin(k2 * r - m2 * theta)
        equations = ['sin(k₁r+m₁θ)', 'sin(k₂r-m₂θ)']
    elif mode == 'rings':
        k = rng.uniform(18, 42)
        texture = np.sin(k * r) + 0.45 * np.sin((k * 0.55) * np.sqrt((x + 0.5) ** 2 + (y - 0.25) ** 2))
        equations = ['concentric rings']
    elif mode == 'checker':
        ax, by = rng.uniform(10, 24), rng.uniform(8, 22)
        texture = np.sin(ax * x) * np.sin(by * y)
        texture = np.sign(texture) * np.sqrt(np.abs(texture))
        texture = cv2.GaussianBlur(texture.astype(np.float32), (0, 0), sigmaX=0.7)
        equations = ['soft checker']
    elif mode == 'woven':
        ax, by = rng.uniform(12, 28), rng.uniform(12, 28)
        texture = np.sin(ax * x) + np.cos(by * y) + 0.55 * np.sin(ax * x) * np.cos(by * y)
        equations = ['sin(ax)+cos(by)+product']
    elif mode == 'contour':
        z = 1.7 * x * x - 1.2 * y * y + 0.9 * x * y + rng.uniform(-0.8, 0.8) * x
        texture = np.sin(rng.uniform(10, 24) * z)
        equations = ['sin(quadratic contour)']
    elif mode == 'microgrid':
        sx, sy = rng.uniform(0.055, 0.12), rng.uniform(0.10, 0.22)
        gx = np.exp(-((np.mod(xx + rng.uniform(0, sx * w), sx * w) / (sx * w)) - 0.5) ** 2 / 0.015)
        gy = np.exp(-((np.mod(yy + rng.uniform(0, sy * h), sy * h) / (sy * h)) - 0.5) ** 2 / 0.018)
        texture = gx + gy - np.median(gx + gy)
        equations = ['periodic microgrid']
    elif mode == 'moire':
        a = rng.uniform(18, 36)
        texture = (
            np.sin(a * (x + 0.10 * y) + rng.uniform(0, 2 * np.pi)) +
            np.sin((a + rng.uniform(0.35, 1.8)) * (x - 0.12 * y) + rng.uniform(0, 2 * np.pi)) +
            0.5 * np.sin((a * 0.7) * (0.3 * x + y))
        )
        equations = ['three-wave moiré']
    elif mode == 'dots':
        ax, by = rng.uniform(14, 26), rng.uniform(14, 26)
        lattice = (np.cos(ax * x) + np.cos(by * y)) * 0.5
        texture = np.clip(lattice, 0.35, 1.0) ** 5
        texture -= texture.mean()
        texture = cv2.GaussianBlur(texture.astype(np.float32), (0, 0), sigmaX=0.45)
        equations = ['cosine dot lattice']
    else:
        texture = np.sin(12 * x + 7 * y)
        equations = ['fallback sine']

    texture = texture.astype(np.float32)
    texture -= np.median(texture)
    texture /= max(np.percentile(np.abs(texture), 98), 1e-6)
    return np.clip(texture, -1, 1), [mode] + equations

def synth_diverse_background(h=78, w=430, seed=0):
    rng = np.random.default_rng(seed)
    palettes = [
        ([238, 240, 232], [205, 220, 245]),
        ([220, 235, 226], [245, 230, 205]),
        ([232, 226, 246], [215, 240, 238]),
        ([245, 236, 218], [224, 235, 250]),
        ([216, 232, 244], [241, 241, 225]),
    ]
    c0, c1 = [np.array(x, np.float32) for x in palettes[int(rng.integers(0, len(palettes)))]]
    yy, xx = np.mgrid[0:h, 0:w]
    mix = (xx / max(w - 1, 1)) * rng.uniform(0.15, 0.85) + (yy / max(h - 1, 1)) * rng.uniform(0.05, 0.25)
    img = c0 * (1 - mix[..., None]) + c1 * mix[..., None]
    mode = SYNTH_BACKGROUND_MODES[int(seed) % len(SYNTH_BACKGROUND_MODES)]
    math_texture, equations = random_math_texture(h, w, rng, mode=mode)
    # Strong enough to inspect in previews; panel normalization later prevents this from dominating.
    tint = rng.normal(0, rng.uniform(18, 42), size=3)
    img += math_texture[..., None] * tint
    for _ in range(int(rng.integers(3, 9))):
        angle = rng.uniform(0, np.pi)
        freq = rng.uniform(0.015, 0.075)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(1.5, 8.0)
        wave = np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * freq + phase)
        channel = int(rng.integers(0, 3))
        img[..., channel] += amp * wave
    if rng.random() < 0.8:
        step = int(rng.integers(9, 24))
        color = rng.normal(0, rng.uniform(2, 7), size=3)
        img[:, ::step] += color
        if rng.random() < 0.55:
            img[::max(6, step // 2), :] -= color * rng.uniform(0.4, 0.9)
    if rng.random() < 0.25:
        # Subtle security-line overlay; keep it weak so it does not collapse every sample into diagonal stripes.
        orientation = rng.choice(['vertical', 'horizontal', 'diagonal'])
        spacing = int(rng.integers(18, 42))
        color = tuple(float(x) for x in rng.normal(0, 0.8, size=3))
        if orientation == 'vertical':
            for x0 in range(0, w, spacing):
                cv2.line(img, (x0, 0), (x0, h - 1), color, 1, cv2.LINE_AA)
        elif orientation == 'horizontal':
            for y0 in range(0, h, spacing):
                cv2.line(img, (0, y0), (w - 1, y0), color, 1, cv2.LINE_AA)
        else:
            for k in range(-h, w, spacing):
                cv2.line(img, (max(0, k), max(0, -k)), (min(w - 1, k + h), min(h - 1, h + k)), color, 1, cv2.LINE_AA)
    img += rng.normal(0, rng.uniform(0.8, 3.2), size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)

def add_edge_text_decoys(img, rng, font, color, thickness):
    """Add partial text snippets at crop edges — simulating neighboring fields.
    
    Text is placed OUTSIDE the main content area, only partially visible
    at the very edge of the crop. Never overlaps with the main text.
    """
    h, w = img.shape[:2]
    if h < 30 or w < 80:
        return img
    
    # Find where main text is so we avoid it
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)
    ink = ((gray < local - 22) & (gray < 175)) | (gray < 80)
    if ink.any():
        ys, xs = np.where(ink)
        text_y0, text_y1 = int(ys.min()), int(ys.max())
        text_x0, text_x1 = int(xs.min()), int(xs.max())
    else:
        text_y0, text_y1 = h // 4, 3 * h // 4
        text_x0, text_x1 = w // 4, 3 * w // 4

    snippets = [
        rng.choice(SYNTH_FIELD_LABELS[rng.choice(list(SYNTH_FIELD_LABELS.keys()))]),
        synth_random_date(rng),
    ]
    text = str(rng.choice(snippets))
    
    try:
        decoy_font = choose_synth_font(rng, size=max(12, int(font.size * rng.uniform(0.7, 1.0))), text=text)
        c = int(np.clip(color + rng.integers(-12, 28), 20, 130))
        tw, th = pil_text_size(text, decoy_font)
        
        # Place at edge — mostly OUTSIDE the crop, only a sliver visible
        # Never within the main text bounding box
        edge = rng.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top' and text_y0 > th // 2:
            # Place above main text, partially clipped at top edge
            px = int(rng.uniform(0, max(1, w - tw)))
            py = int(rng.uniform(-th * 0.7, -th * 0.2))  # mostly above crop
        elif edge == 'bottom' and (h - text_y1) > th // 2:
            px = int(rng.uniform(0, max(1, w - tw)))
            py = int(rng.uniform(h - th * 0.3, h))  # mostly below crop
        elif edge == 'left' and text_x0 > tw // 3:
            px = int(rng.uniform(-tw * 0.8, -tw * 0.3))  # mostly left of crop
            py = int(rng.uniform(0, max(1, h - th)))
        elif edge == 'right' and (w - text_x1) > tw // 3:
            px = int(rng.uniform(w - tw * 0.3, w))  # mostly right of crop
            py = int(rng.uniform(0, max(1, h - th)))
        else:
            return img  # no safe edge to place decoy
        
        pil_put_text(img, text, (px, py), decoy_font, c, blur=False)
    except (ValueError, IndexError, OSError):
        pass
    return img
def _synth_ink_mask(rendered, background):
    """Detect ink pixels by diffing rendered image against its clean background."""
    diff = np.abs(rendered.astype(np.float32) - background.astype(np.float32)).max(axis=2)
    return diff > 6.0

def _synth_clean_background(rendered, ink_mask):
    """Estimate clean background by inpainting over ink pixels."""
    mask_u8 = (ink_mask.astype(np.uint8) * 255)
    mask_u8 = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(rendered, mask_u8, 3, cv2.INPAINT_TELEA)

def _synth_text_effect_block(ink_mask, pad_x=4, pad_y=3):
    """Tight bounding box around ink pixels with padding."""
    if not ink_mask.any():
        h, w = ink_mask.shape
        return 0, 0, w, h
    ys, xs = np.where(ink_mask)
    h, w = ink_mask.shape
    return (
        max(0, int(xs.min()) - pad_x),
        max(0, int(ys.min()) - pad_y),
        min(w, int(xs.max()) + 1 + pad_x),
        min(h, int(ys.max()) + 1 + pad_y),
    )

def _synth_pick_subspan(ink_mask, x0, y0, x1, y1, rng, frac_range=(0.20, 0.50)):
    """Pick a contiguous ink sub-window that is a STRICT subset of the text.

    Guarantees some glyphs remain outside the span, so they stay on the baseline as
    the reference against which the shifted span is visibly mal-aligned.
    """
    band_ink = ink_mask[y0:y1, x0:x1]
    if not band_ink.any():
        return None
    cols = np.flatnonzero(band_ink.any(axis=0))
    if len(cols) < 8:
        return None
    ink_x0, ink_x1 = int(cols.min()), int(cols.max()) + 1
    ink_w = ink_x1 - ink_x0
    span_w = int(round(ink_w * rng.uniform(*frac_range)))
    span_w = max(4, min(span_w, ink_w - 4))   # keep >=4px of neighbors outside the span
    if span_w < 4 or span_w >= ink_w:
        return None
    start = int(rng.integers(ink_x0, ink_x1 - span_w))
    return (x0 + start, x0 + start + span_w)

def _synth_smooth_row_shift(out, bg, ink_mask, x0, y0, x1, y1, rng):
    """Smooth per-row horizontal shift of the text ink (the real train/test row-shift tamper).

    Each horizontal scanline of ink is displaced along x. The displacement varies SMOOTHLY
    across rows (e.g. 5,4,3,2... px) with two hard constraints that match captured tampering:
      * |shift[row] - shift[row-1]| <= 1 px   -> no sharp jumps (never 10px -> 1px)
      * velocity changes slowly                -> no sharp curves / reversals (never +5 -> -5)
    Direction is arbitrary (left or right). Background pixels stay put; only ink moves.
    """
    width = x1 - x0
    rows = list(range(y0, y1))
    n_rows = len(rows)
    if width < 6 or n_rows < 3:
        return
    max_shift = int(rng.integers(6, 11))                     # peak displacement 6..10 px (always clearly visible)
    # Momentum walk: per-row delta is the velocity (|v|<=1 -> no sharp jump);
    # velocity drifts slowly (-0.3..0.3 -> low curvature, no sharp reversal).
    shift = float(rng.integers(-max_shift, max_shift + 1))
    velocity = float(rng.uniform(-1.0, 1.0))
    shifts = np.empty(n_rows, dtype=int)
    for i in range(n_rows):
        shifts[i] = int(round(shift))
        velocity = float(np.clip(velocity + rng.uniform(-0.3, 0.3), -1.0, 1.0))
        shift = float(np.clip(shift + velocity, -max_shift, max_shift))
    for idx, y in enumerate(rows):
        shift_px = int(shifts[idx])
        if shift_px == 0:
            continue
        row_mask = ink_mask[y, x0:x1]
        if not row_mask.any():
            continue
        orig = out[y, x0:x1].copy()
        fill = bg[y, x0:x1]
        out[y, x0:x1][row_mask] = fill[row_mask]            # clear ink at its original x
        moved = np.where(row_mask)[0]
        new_pos = moved + shift_px
        valid = (new_pos >= 0) & (new_pos < width)
        out[y, x0 + new_pos[valid]] = orig[moved[valid]]    # repaint ink at shifted x

def _synth_copy_move(out, bg, ink_mask, x0, y0, x1, y1, rng):
    """Copy-move: duplicate a small character span and paste it at a shifted location.

    DocTamper-style copy-move = "shifting the spatial locations of texts within images".
    Leaves the source glyphs intact and stamps a duplicate nearby -> repeated-character cue.
    """
    span = _synth_pick_subspan(ink_mask, x0, y0, x1, y1, rng, frac_range=(0.16, 0.34))
    if span is None:
        return
    sx0, sx1 = span
    span_w = sx1 - sx0
    sy0 = max(0, y0 - 4)
    sy1 = min(out.shape[0], y1 + 4)
    span_h = sy1 - sy0
    span_mask = ink_mask[sy0:sy1, sx0:sx1]
    if not span_mask.any():
        return
    glyphs = out[sy0:sy1, sx0:sx1].copy()
    # Shift roughly one span-width left/right, then clamp so the paste stays inside the field.
    direction = int(rng.choice([-1, 1]))
    dx = direction * int(rng.integers(max(3, span_w // 2), span_w + 5))
    dy = int(rng.choice([-2, -1, 0, 1, 2]))
    dest_x0 = int(np.clip(sx0 + dx, x0, max(x0, x1 - span_w)))
    # If clamping landed us back on the source, force a non-trivial offset the other way.
    if abs(dest_x0 - sx0) < 3:
        dest_x0 = int(np.clip(sx0 - dx, x0, max(x0, x1 - span_w)))
    dest_y0 = int(np.clip(sy0 + dy, 0, out.shape[0] - span_h))
    dest = out[dest_y0:dest_y0 + span_h, dest_x0:dest_x0 + span_w]
    if dest.shape[:2] != span_mask.shape:
        return
    dest[span_mask] = glyphs[span_mask]

def _synth_minimal_row_bg_artifact(out, bg, ink_mask, x0, y0, x1, y1, rng):
    """Background tint + brightness shift + white dots on non-ink pixels, within text region only."""
    rh, rw = y1 - y0, x1 - x0
    if rh <= 0 or rw <= 0:
        return
    local_ink = ink_mask[y0:y1, x0:x1]
    local_bg = ~local_ink
    if not local_bg.any():
        return
    region = out[y0:y1, x0:x1].astype(np.float32)
    blur_sigma = rng.uniform(0.8, 1.8)
    smooth = np.stack([cv2.GaussianBlur(region[:, :, c], (0, 0), sigmaX=blur_sigma) for c in range(3)], axis=2)
    mean_val = smooth.mean(axis=(0, 1), keepdims=True)
    contrast = rng.uniform(0.85, 1.05)
    patch = (smooth - mean_val) * contrast + mean_val
    patch += rng.uniform(-24.0, -10.0)
    tint = np.array([198, 222, 245], dtype=np.float32)
    tint_alpha = rng.uniform(0.20, 0.36)
    patch = patch * (1.0 - tint_alpha) + tint * tint_alpha
    smooth_alpha = rng.uniform(0.40, 0.65)
    blended = (region * (1.0 - smooth_alpha) + patch * smooth_alpha).clip(0, 255).astype(np.uint8)
    out[y0:y1, x0:x1][local_bg] = blended[local_bg]
    if rng.random() < 0.35:
        density = rng.uniform(0.001, 0.006)
        dots = (rng.random((rh, rw)) < density) & local_bg
        if dots.any():
            alpha = rng.uniform(0.40, 0.80)
            white = np.array([255, 255, 255], dtype=np.float32)
            dotted = (out[y0:y1, x0:x1].astype(np.float32) * (1.0 - alpha) + white * alpha).clip(0, 255).astype(np.uint8)
            out[y0:y1, x0:x1][dots] = dotted[dots]

def _synth_black_pixel_contrast(out, bg, ink_mask, x0, y0, x1, y1, rng):
    """Increase or decrease ink pixel contrast against local background."""
    mode = 'increase' if rng.random() < 0.5 else 'decrease'
    strength = rng.uniform(0.35, 0.60)
    ys, xs = np.where(ink_mask[y0:y1, x0:x1])
    if len(xs) == 0:
        return
    yy = ys + y0
    xx = xs + x0
    base = bg[yy, xx].astype(np.float32)
    pix = out[yy, xx].astype(np.float32)
    factor = 1.0 + strength if mode == 'increase' else 1.0 - strength
    out[yy, xx] = (base + (pix - base) * factor).clip(0, 255).astype(np.uint8)

def _synth_non_ink_background_noise(out, bg, ink_mask, x0, y0, x1, y1, rng):
    """Gaussian noise + shifted-copy exchange on background-only pixels."""
    bg_mask = ~ink_mask[y0:y1, x0:x1]
    if not bg_mask.any():
        return
    region = out[y0:y1, x0:x1]
    visibility = rng.uniform(3.0, 9.0)
    std = rng.uniform(4.0, 9.0) * visibility
    noise = rng.normal(0, std, region.shape).astype(np.float32)
    noisy = (region.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    region[bg_mask] = noisy[bg_mask]
    dx = int(rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))
    dy = int(rng.choice([-2, -1, 0, 1, 2]))
    shifted = np.roll(np.roll(region, dy, axis=0), dx, axis=1)
    safe = bg_mask & np.roll(np.roll(bg_mask, dy, axis=0), dx, axis=1)
    if dy > 0: safe[:dy, :] = False
    elif dy < 0: safe[dy:, :] = False
    if dx > 0: safe[:, :dx] = False
    elif dx < 0: safe[:, dx:] = False
    region[safe] = shifted[safe]

def _synth_faint_horizontal_lines(out, bg, ink_mask, x0, y0, x1, y1, rng):
    """Colored horizontal lines through text rows."""
    height = y1 - y0
    if height <= 0:
        return
    text_rows = np.flatnonzero(ink_mask[y0:y1, x0:x1].any(axis=1)) + y0
    if len(text_rows) == 0:
        return
    visibility = rng.uniform(1.2, 3.0)
    n_lines = max(1, int(round(int(rng.integers(5, 9)) * visibility)))
    colors = [
        ('green', np.array([190, 235, 205], dtype=np.float32)),
        ('orange', np.array([245, 215, 170], dtype=np.float32)),
        ('blue', np.array([185, 215, 245], dtype=np.float32)),
    ]
    for _ in range(n_lines):
        line_width = max(1, int(round(int(rng.integers(2, 5)) * min(max(visibility, 0.25), 2.5))))
        center_y = int(rng.choice(text_rows))
        y_line = max(y0, min(max(y0, y1 - line_width), center_y + int(rng.integers(-2, 3))))
        _, color_rgb = colors[int(rng.integers(0, len(colors)))]
        color_line = color_rgb * rng.uniform(0.85, 1.15)
        color_line = color_line.clip(0, 255)
        alpha = min(0.85, rng.uniform(0.30, 0.45) * visibility)
        for yy in range(y_line, min(y_line + line_width, y1)):
            mask = ~ink_mask[yy, x0:x1]
            if not mask.any():
                continue
            row = out[yy, x0:x1]
            blended = (row.astype(np.float32) * (1.0 - alpha) + color_line * alpha).clip(0, 255).astype(np.uint8)
            row[mask] = blended[mask]

def _synth_field_static_noise(out, bg, ink_mask, x0, y0, x1, y1, rng):
    """TV-static speckle noise."""
    region = out[y0:y1, x0:x1]
    if region.size == 0:
        return
    h, w = region.shape[:2]
    density = rng.uniform(0.07, 0.16)
    speckles = rng.random((h, w)) < density
    if not speckles.any():
        return
    strength = rng.uniform(0.30, 0.60)
    static = rng.integers(0, 256, (h, w, 1), dtype=np.uint8)
    static = np.repeat(static, 3, axis=2)
    alpha = np.full((h, w, 1), strength, dtype=np.float32)
    alpha[ink_mask[y0:y1, x0:x1]] *= 0.35
    blended = (region.astype(np.float32) * (1.0 - alpha) + static.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    region[speckles] = blended[speckles]

def _synth_light_text_blur(out, bg, ink_mask, x0, y0, x1, y1, rng):
    """Selective gaussian blur on ink-masked pixels only."""
    region = out[y0:y1, x0:x1]
    if region.size == 0:
        return
    mask = ink_mask[y0:y1, x0:x1].copy()
    if not mask.any():
        return
    from scipy import ndimage as _ndi
    mask = _ndi.binary_dilation(mask, structure=np.ones((2, 2)), iterations=1)
    visibility = rng.uniform(1.8, 4.0)
    radius = rng.uniform(0.30, 0.70) * max(0.1, visibility)
    strength = min(1.0, rng.uniform(0.45, 0.75) * visibility)
    blurred = cv2.GaussianBlur(region, (0, 0), sigmaX=max(0.1, radius))
    blended = (region.astype(np.float32) * (1.0 - strength) + blurred.astype(np.float32) * strength).clip(0, 255).astype(np.uint8)
    region[mask] = blended[mask]

_REPRINT_ACCENTS = {
    'a': 'aaaaa', 'e': 'eeeee', 'i': 'iiii', 'o': 'ooooo', 'u': 'uuuu',
}
_REPRINT_ACCENTS = {
    'a': '\u00e0\u00e2\u00e4\u00e3\u00e1', 'e': '\u00e9\u00e8\u00ea\u00eb\u0115',
    'i': '\u00ee\u00ef\u00ed\u012d', 'o': '\u00f6\u00f4\u00f5\u00f3\u014f',
    'u': '\u00fb\u00fc\u00fa\u016d', 'A': '\u00c0\u00c2\u00c4', 'E': '\u00c9\u00c8\u00ca',
    'O': '\u00d6\u00d4\u00d5', 'U': '\u00db\u00dc',
}
def _reprint_accentize(s, rng, p=0.22):
    out = []
    for ch in s:
        acc = _REPRINT_ACCENTS.get(ch)
        out.append(str(rng.choice(list(acc))) if (acc and rng.random() < p) else ch)
    return ''.join(out)

def _reprint_new_text(rng, approx_chars):
    """New value text matching real captured tampers: accented names, Latin dates, digit runs."""
    kind = rng.choice(['name', 'date', 'digits'], p=[0.5, 0.25, 0.25])
    if kind == 'date':
        return synth_random_date(rng)
    if kind == 'digits':
        n = int(max(3, min(14, approx_chars)))
        return ''.join(str(int(d)) for d in rng.integers(0, 10, size=n))
    return _reprint_accentize(synth_random_name(rng), rng)

def _synth_clean_text_reprint(out, bg, ink_mask, x0, y0, x1, y1, rng):
    """Guinea-style patch-and-rewrite tamper.

    Real tamper signature (from forensic panel analysis):
    - The LABEL text ("Date de naissance:", "Prénoms:") stays on ORIGINAL background
    - Only the VALUE portion gets a rectangular PATCH of slightly different background
    - The patch creates a visible discontinuity in L-residual (luminance shift)
    - New text is rendered on the patch in a slightly different font/weight
    - The patch has soft but visible edges (feathered seam)

    This is the primary forensic tell: within one field crop, the left portion
    (label) has original background while the right portion (value) sits on a
    visibly different background rectangle.
    """
    band = ink_mask[y0:y1, x0:x1]
    if not band.any():
        return
    cols = np.flatnonzero(band.any(axis=0))
    rows_ink = np.flatnonzero(band.any(axis=1))
    if len(cols) < 8 or len(rows_ink) < 4:
        return
    ink_x0, ink_x1 = x0 + int(cols.min()), x0 + int(cols.max()) + 1
    ink_y0, ink_y1 = y0 + int(rows_ink.min()), y0 + int(rows_ink.max()) + 1
    gh, gw = ink_y1 - ink_y0, ink_x1 - ink_x0
    if gh < 6 or gw < 12:
        return
    h_img, w_img = out.shape[:2]

    # --- Find the VALUE portion (right part of the text) ---
    # In "Label: Value" fields, the value starts roughly at the midpoint or after ':'
    # For value-only fields, patch the whole thing or a trailing portion
    if gw > 80 and rng.random() < 0.65:
        # Partial: patch only the right portion (the "value" part)
        split_frac = rng.uniform(0.3, 0.6)  # where label ends and value begins
        val_x0 = ink_x0 + int(gw * split_frac)
        val_x1 = ink_x1
    elif gw > 40 and rng.random() < 0.3:
        # Just one word in the middle/end
        sw = int(max(20, gw * rng.uniform(0.2, 0.4)))
        val_x1 = ink_x1
        val_x0 = max(ink_x0, val_x1 - sw)
    else:
        # Full text (short fields or value-only)
        val_x0, val_x1 = ink_x0, ink_x1

    # --- Create the PATCH rectangle (only under the value) ---
    pad_x = int(rng.integers(6, 14))
    pad_y = int(rng.integers(4, 10))
    margin = int(rng.integers(6, 14))

    patch_x0 = max(margin, val_x0 - pad_x)
    patch_y0 = max(margin, ink_y0 - pad_y)
    patch_x1 = min(w_img - margin, val_x1 + pad_x)
    patch_y1 = min(h_img - margin, ink_y1 + pad_y)
    pw, ph = patch_x1 - patch_x0, patch_y1 - patch_y0
    if pw < 15 or ph < 8:
        return

    # --- Patch color: slightly shifted luminance (creates L-residual signature) ---
    local_bg = out[patch_y0:patch_y1, patch_x0:patch_x1]
    local_ink_mask = ink_mask[patch_y0:patch_y1, patch_x0:patch_x1]
    bg_pixels = local_bg[~local_ink_mask] if (~local_ink_mask).any() else local_bg.reshape(-1, 3)
    bg_color = np.median(bg_pixels, axis=0).astype(np.float32)

    # Luminance shift: the key forensic signature
    # Real patches are typically slightly brighter or slightly darker
    lum_shift = rng.uniform(5, 20) * rng.choice([-1, 1])
    # Small color temperature shift too
    color_shift = rng.uniform(-6, 6, size=3).astype(np.float32)
    patch_color = np.clip(bg_color + lum_shift + color_shift, 0, 255)

    # Fill the patch with feathered edges
    patch_region = out[patch_y0:patch_y1, patch_x0:patch_x1].astype(np.float32)
    alpha = np.ones((ph, pw), np.float32)
    feather = int(rng.integers(2, 5))
    for f in range(feather):
        a = (f + 1) / (feather + 1)
        alpha[f, :] *= a; alpha[-(f+1), :] *= a
        alpha[:, f] *= a; alpha[:, -(f+1)] *= a
    alpha = alpha[:, :, np.newaxis]
    blended = patch_region * (1 - alpha * 0.85) + patch_color * (alpha * 0.85)
    out[patch_y0:patch_y1, patch_x0:patch_x1] = np.clip(blended, 0, 255).astype(np.uint8)

    # --- Blend transparent distortion overlay into the patch background ---
    # Simulates mismatched security texture: semi-transparent horizontal wave
    # pattern blended into existing background, then text rendered on top.
    if rng.random() < 0.75:
        pr = out[patch_y0:patch_y1, patch_x0:patch_x1].astype(np.float32)
        bg_in_patch = ~ink_mask[patch_y0:patch_y1, patch_x0:patch_x1]
        yy, xx = np.meshgrid(np.arange(ph), np.arange(pw), indexing='ij')
        # Wavy horizontal distortion
        freq = rng.uniform(0.12, 0.45)
        phase = rng.uniform(0, 2 * np.pi)
        wave_mod = rng.uniform(0.008, 0.03)
        wave = np.sin(2 * np.pi * freq * yy + wave_mod * xx + phase)
        # Second harmonic for complexity
        wave += 0.4 * np.sin(2 * np.pi * freq * 1.7 * yy + phase * 0.7)
        wave = wave / wave.max()  # normalize to [-1, 1]
        # Transparent blend: shift existing pixels toward bright/dark based on wave
        opacity = rng.uniform(0.06, 0.20)
        shift = wave * rng.uniform(15, 40)  # luminance modulation
        for ch in range(3):
            pr[:, :, ch][bg_in_patch] = pr[:, :, ch][bg_in_patch] + shift[bg_in_patch] * opacity
        out[patch_y0:patch_y1, patch_x0:patch_x1] = np.clip(pr, 0, 255).astype(np.uint8)

    # --- Inpaint original ink within the value span ---
    span_ink = ink_mask[ink_y0:ink_y1, val_x0:val_x1]
    if span_ink.any():
        m = np.zeros(out.shape[:2], np.uint8)
        m[ink_y0:ink_y1, val_x0:val_x1][span_ink] = 255
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), 1)
        cleaned = cv2.inpaint(out, m, 3, cv2.INPAINT_TELEA)
        # Replace only within the patch area
        mask_in_patch = m[patch_y0:patch_y1, patch_x0:patch_x1] > 0
        out[patch_y0:patch_y1, patch_x0:patch_x1][mask_in_patch] =             cleaned[patch_y0:patch_y1, patch_x0:patch_x1][mask_in_patch]
        # Tint the inpainted area with patch color
        inpainted = out[patch_y0:patch_y1, patch_x0:patch_x1].astype(np.float32)
        tint_alpha = rng.uniform(0.3, 0.6)
        inpainted[mask_in_patch] = inpainted[mask_in_patch] * (1 - tint_alpha) + patch_color * tint_alpha
        out[patch_y0:patch_y1, patch_x0:patch_x1] = np.clip(inpainted, 0, 255).astype(np.uint8)

    # --- Render new text fitted within the patch ---
    ink_color_gray = 45
    orig_ink_pixels = out[ink_y0:ink_y1, val_x0:val_x1]
    orig_ink_mask = ink_mask[ink_y0:ink_y1, val_x0:val_x1]
    reddish = False
    if orig_ink_mask.any():
        iv = orig_ink_pixels[orig_ink_mask]
        if len(iv) > 0:
            ink_color_gray = int(np.clip(np.percentile(iv, 20), 20, 80))
            mean_rgb = iv.astype(np.float32).mean(axis=0)
            reddish = (mean_rgb[0] > mean_rgb[1] + 25) and (mean_rgb[0] > mean_rgb[2] + 25)

    # Text area = inside patch with small margin
    tx0 = patch_x0 + max(3, pad_x // 3)
    tx1 = patch_x1 - max(3, pad_x // 3)
    ty0 = patch_y0 + max(2, pad_y // 3)
    ty1 = patch_y1 - max(2, pad_y // 3)
    aw, ah = tx1 - tx0, ty1 - ty0
    if aw < 15 or ah < 6:
        return

    # Generate new text
    approx_chars = int(max(2, aw / max(6.0, gh * 0.55)))
    kind = rng.choice(['name', 'date', 'digits'], p=[0.5, 0.25, 0.25])
    if kind == 'date':
        txt = synth_random_date(rng)
    elif kind == 'digits':
        n = int(max(3, min(14, approx_chars)))
        txt = ''.join(str(int(d)) for d in rng.integers(0, 10, size=n))
    else:
        txt = _reprint_accentize(synth_random_name(rng), rng)

    # Fit font
    for fsize in range(int(max(8, gh * 1.1)), 7, -1):
        font = choose_synth_font(rng, fsize, text=txt, italic_p=0.12)
        tw, th = pil_text_size(txt, font)
        if tw <= aw and th <= ah:
            break
    else:
        return

    px = tx0 + int(rng.integers(0, max(1, aw - tw + 1)))
    py = ty0 + max(0, (ah - th) // 2) + int(rng.integers(-1, 2))

    snap = out.copy()
    pil_put_text(out, txt, (px, py), font, ink_color_gray, blur=False)
    if reddish:
        drawn = np.any(out != snap, axis=2)
        out[drawn] = np.array([165, 35, 45], np.uint8)

def _synth_cool_blue_bg_filter(out, bg, ink_mask, x0, y0, x1, y1, rng):
    """Cool/light-blue color filter on the background within the text bounding box."""
    h, w = out.shape[:2]
    bx0, by0, bx1, by1 = x0, y0, x1, y1
    region = out[by0:by1, bx0:bx1].astype(np.float32)
    bg_mask = ~ink_mask[by0:by1, bx0:bx1]
    if not bg_mask.any():
        return
    presets = [
        np.array([210, 225, 248], np.float32),
        np.array([195, 220, 245], np.float32),
        np.array([215, 235, 250], np.float32),
        np.array([200, 230, 240], np.float32),
        np.array([220, 230, 248], np.float32),
    ]
    tint = presets[int(rng.integers(0, len(presets)))]
    alpha = rng.uniform(0.20, 0.55)
    tinted = region * (1.0 - alpha) + tint * alpha
    bright = rng.uniform(-15.0, 18.0)
    tinted = np.clip(tinted + bright, 0, 255).astype(np.uint8)
    out[by0:by1, bx0:bx1][bg_mask] = tinted[bg_mask]
    if rng.random() < 0.65:
        edge_color = int(np.clip(rng.uniform(180, 230), 0, 255))
        cv2.rectangle(out, (bx0, by0), (bx1 - 1, by1 - 1), (edge_color, edge_color, edge_color + 10), 1)
    if rng.random() < 0.35:
        ink_local = ink_mask[by0:by1, bx0:bx1]
        if ink_local.any():
            shift = rng.uniform(-22, 12)
            pixels = out[by0:by1, bx0:bx1].astype(np.float32)
            pixels[ink_local] = np.clip(pixels[ink_local] + shift, 0, 255)
            out[by0:by1, bx0:bx1][ink_local] = pixels[ink_local].astype(np.uint8)


def _synth_diffusion_spread(out, bg, ink_mask, x0, y0, x1, y1, rng, strength=None, thickness=None):
    """MNIST-style stroke irregularity with smooth alpha-blended segment dilation."""
    band = ink_mask[y0:y1, x0:x1]
    if not band.any():
        return
    rh, rw = y1 - y0, x1 - x0
    if rh < 6 or rw < 10:
        return
    if strength is None:
        strength = float(rng.uniform(0.1, 0.7))
    if thickness is None:
        thickness = float(rng.uniform(0.0, 0.6))
    strength = float(np.clip(strength, 0.0, 0.8))
    thickness = float(np.clip(thickness, 0.0, 0.8))
    if strength < 0.01 and thickness < 0.01:
        return

    region = out[y0:y1, x0:x1]
    bg_region = bg[y0:y1, x0:x1]
    ink_pixels = region[band]
    ink_color = np.percentile(ink_pixels, 25, axis=0).astype(np.uint8)

    # Variation map
    raw_noise = rng.standard_normal((rh, rw)).astype(np.float32)
    sigma = rng.uniform(8, 18)
    var_map = cv2.GaussianBlur(raw_noise, (0, 0), sigma)
    vmin, vmax = var_map.min(), var_map.max()
    if vmax - vmin > 1e-6:
        var_map = (var_map - vmin) / (vmax - vmin)

    new_ink = band.copy()

    # Smooth alpha-blended segment dilation
    if thickness > 0.001:
        dilated_1 = cv2.dilate(band.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
        new_edge_1 = dilated_1 & ~band
        blob_noise = rng.standard_normal((rh, rw)).astype(np.float32)
        blobs = cv2.GaussianBlur(blob_noise, (0, 0), rng.uniform(4, 8))
        bmin, bmax = blobs.min(), blobs.max()
        if bmax - bmin > 1e-6:
            blobs = (blobs - bmin) / (bmax - bmin)
        alpha_map = (blobs * thickness).clip(0, 1)
        alpha_map = cv2.GaussianBlur(alpha_map, (5, 5), 1.5)
        mask = new_edge_1 & (alpha_map > 0.05)
        if mask.any():
            a = alpha_map[mask].reshape(-1, 1).astype(np.float32)
            region[mask] = np.clip(
                region[mask].astype(np.float32) * (1 - a) + ink_color.astype(np.float32) * a,
                0, 255
            ).astype(np.uint8)
        if thickness > 0.5:
            dilated_2 = cv2.dilate(dilated_1.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
            new_edge_2 = dilated_2 & ~dilated_1
            hot_alpha = ((blobs - 0.5) * 2 * (thickness - 0.3)).clip(0, 1)
            hot_alpha = cv2.GaussianBlur(hot_alpha, (5, 5), 1.5)
            mask2 = new_edge_2 & (hot_alpha > 0.05)
            if mask2.any():
                a2 = hot_alpha[mask2].reshape(-1, 1).astype(np.float32)
                region[mask2] = np.clip(
                    region[mask2].astype(np.float32) * (1 - a2) + ink_color.astype(np.float32) * a2,
                    0, 255
                ).astype(np.uint8)
    # Skip clear+repaint — alpha blending already applied above
    new_ink = band.copy()

    # Erosion for thinning
    if strength > 0.3:
        thin_thresh = -0.7 + 0.4 * strength
        var_map_signed = var_map * 2 - 1
        thin_mask = var_map_signed < thin_thresh
        if thin_mask.any():
            eroded = cv2.erode(band.astype(np.uint8), np.ones((2, 2), np.uint8), 1).astype(bool)
            removed = band & thin_mask & ~eroded
            region[removed] = bg_region[removed]

    # Per-row jitter
    if strength > 0.1:
        jitter_amplitude = strength * 1.8
        jitter_freq = rng.uniform(0.05, 0.15)
        phase = rng.uniform(0, 2 * np.pi)
        for row in range(rh):
            shift = int(round(jitter_amplitude * np.sin(2 * np.pi * jitter_freq * row + phase)))
            if shift != 0:
                region[row] = np.roll(region[row], shift, axis=0)
                if shift > 0:
                    region[row, :shift] = bg_region[row, :shift]
                else:
                    region[row, shift:] = bg_region[row, shift:]


def _synth_bg_color_morph(out, bg, ink_mask, x0, y0, x1, y1, rng):
    """Background color morph: shift/boost existing background pixel colors
    in a rectangular region while preserving text and pattern structure.

    Like looking through tinted glass — the underlying pattern (guilloche lines,
    security features) stays but colors shift. Green becomes teal, dark patterns
    shift to brown, light areas get warmer/cooler.

    Only background pixels are affected. Text/ink stays untouched.
    """
    band = ink_mask[y0:y1, x0:x1]
    rh, rw = y1 - y0, x1 - x0
    if rh < 6 or rw < 10:
        return
    h_img, w_img = out.shape[:2]

    # Pick the morph rectangle (value portion or partial span)
    cols = np.flatnonzero(band.any(axis=0))
    if len(cols) < 5:
        return
    ink_x0 = x0 + int(cols.min()); ink_x1 = x0 + int(cols.max()) + 1
    gw = ink_x1 - ink_x0
    
    if gw > 60 and rng.random() < 0.6:
        split = rng.uniform(0.3, 0.55)
        rx0 = ink_x0 + int(gw * split)
        rx1 = ink_x1
    else:
        rx0, rx1 = ink_x0, ink_x1

    rows_ink = np.flatnonzero(band.any(axis=1))
    if len(rows_ink) < 3:
        return
    ink_y0 = y0 + int(rows_ink.min()); ink_y1 = y0 + int(rows_ink.max()) + 1

    # Rectangle with padding and margin
    pad_x = int(rng.integers(6, 14)); pad_y = int(rng.integers(4, 10))
    margin = int(rng.integers(5, 12))
    mx0 = max(margin, rx0 - pad_x)
    my0 = max(margin, ink_y0 - pad_y)
    mx1 = min(w_img - margin, rx1 + pad_x)
    my1 = min(h_img - margin, ink_y1 + pad_y)
    mw, mh = mx1 - mx0, my1 - my0
    if mw < 15 or mh < 8:
        return

    # Work in float LAB for perceptual color shifts
    region = out[my0:my1, mx0:mx1].copy()
    bg_mask = ~ink_mask[my0:my1, mx0:mx1]
    if not bg_mask.any():
        return
    
    # Convert to LAB
    lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]

    # --- Color morph: per-pixel transform based on existing color ---
    # Luminance shift: slight brighten or darken
    L_shift = rng.uniform(-12, 12)
    L[bg_mask] = np.clip(L[bg_mask] + L_shift, 0, 255)

    # Boost/shift chrominance: push existing colors in a random direction
    # This preserves the pattern structure but changes the hue
    a_shift = rng.uniform(-15, 15)  # green-red axis
    b_shift = rng.uniform(-15, 15)  # blue-yellow axis
    A[bg_mask] = np.clip(A[bg_mask] + a_shift, 0, 255)
    B[bg_mask] = np.clip(B[bg_mask] + b_shift, 0, 255)

    # Boost saturation of existing colors (makes patterns more vivid)
    sat_boost = rng.uniform(0.8, 1.8)
    a_centered = A[bg_mask] - 128
    b_centered = B[bg_mask] - 128
    A[bg_mask] = np.clip(a_centered * sat_boost + 128, 0, 255)
    B[bg_mask] = np.clip(b_centered * sat_boost + 128, 0, 255)

    # Boost contrast of dark patterns (makes guilloche lines stand out more)
    if rng.random() < 0.6:
        local_mean = cv2.GaussianBlur(L, (0, 0), 8)[bg_mask]
        contrast = rng.uniform(1.1, 1.6)
        L[bg_mask] = np.clip((L[bg_mask] - local_mean) * contrast + local_mean, 0, 255)

    lab[:,:,0] = L; lab[:,:,1] = A; lab[:,:,2] = B
    morphed = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Feather the edges for smooth transition
    alpha = np.ones((mh, mw), np.float32)
    feather = int(rng.integers(3, 6))
    for f in range(feather):
        a = (f + 1) / (feather + 1)
        alpha[f, :] *= a; alpha[-(f+1), :] *= a
        alpha[:, f] *= a; alpha[:, -(f+1)] *= a

    # Apply only to background pixels with feathering
    for ch in range(3):
        orig_ch = region[:, :, ch].astype(np.float32)
        morph_ch = morphed[:, :, ch].astype(np.float32)
        blended = orig_ch * (1 - alpha) + morph_ch * alpha
        # Only apply where bg
        result = orig_ch.copy()
        result[bg_mask] = blended[bg_mask]
        out[my0:my1, mx0:mx1, ch] = np.clip(result, 0, 255).astype(np.uint8)

    # --- Uneven char body extension + slight deformation ---
    # Dilate random segments of ink, keeping new pixels CONNECTED to the char body.
    # Then slightly shift a few scanlines to deform the shape.
    morph_ink = ink_mask[my0:my1, mx0:mx1]
    if morph_ink.any():
        morph_region = out[my0:my1, mx0:mx1]
        # Pick random blob regions where chars will extend
        blob = rng.standard_normal((mh, mw)).astype(np.float32)
        blob = cv2.GaussianBlur(blob, (0, 0), rng.uniform(4, 7))
        bmin, bmax = blob.min(), blob.max()
        if bmax - bmin > 1e-6:
            blob = (blob - bmin) / (bmax - bmin)
        # Dilate only ink pixels within active blob zones (top 15-30%)
        thresh = rng.uniform(0.70, 0.85)
        active = blob > thresh
        active_ink = morph_ink & active
        if active_ink.any():
            # Dilate these connected ink segments by 1px
            grown = cv2.dilate(active_ink.astype(np.uint8), np.ones((3,3), np.uint8), 1).astype(bool)
            new_pixels = grown & ~morph_ink
            # Paint new pixels with nearest ink neighbor color (connected extension)
            if new_pixels.any():
                # Use distance transform to find nearest ink pixel for each new pixel
                dist_map = cv2.distanceTransform((~morph_ink).astype(np.uint8), cv2.DIST_L2, 3)
                # For new pixels, grab color from the nearest ink pixel via dilation
                color_source = morph_region.copy()
                # Spread ink colors outward by dilating the color image masked to ink
                for _ in range(2):
                    expanded = cv2.dilate(
                        np.where(morph_ink[:,:,np.newaxis], color_source, 0).astype(np.uint8),
                        np.ones((3,3), np.uint8), 1
                    )
                    ink_expanded = cv2.dilate(morph_ink.astype(np.uint8), np.ones((3,3), np.uint8), 1).astype(bool)
                    # Fill new pixels with expanded colors
                    fill_mask = new_pixels & ink_expanded & ~morph_ink
                    morph_region[fill_mask] = expanded[fill_mask]
        # Slight deformation: shift a few random rows by ±1px
        n_shift = int(mh * rng.uniform(0.05, 0.15))
        shift_rows = rng.choice(mh, size=min(n_shift, mh), replace=False)
        for row in shift_rows:
            if morph_ink[row].any():
                dx = int(rng.choice([-3, -2, -1, 1, 2, 3]))
                morph_region[row] = np.roll(morph_region[row], dx, axis=0)

    # --- Background color bleed into morph zone at random spots ---
    # At some pixels along the morph rectangle edge, let the ORIGINAL
    # (unmorphed) background color leak in by shifting/copying a few
    # adjacent outside pixels into the morph zone. Creates an uneven
    # boundary where the patch doesn't perfectly cover — like a patch
    # that's slightly misaligned in spots.
    if bg_mask.any():
        morph_region = out[my0:my1, mx0:mx1]
        # Sample random spots along the 4 edges of the morph rectangle
        n_bleeds = int(rng.integers(5, 18))
        for _ in range(n_bleeds):
            edge = int(rng.integers(0, 4))
            if edge == 0:  # top
                bx = int(rng.integers(0, mw))
                by = 0
                dy, dx_dir = 1, 0
            elif edge == 1:  # bottom
                bx = int(rng.integers(0, mw))
                by = mh - 1
                dy, dx_dir = -1, 0
            elif edge == 2:  # left
                bx = 0
                by = int(rng.integers(0, mh))
                dy, dx_dir = 0, 1
            else:  # right
                bx = mw - 1
                by = int(rng.integers(0, mh))
                dy, dx_dir = 0, -1
            # Copy a small streak (2-5px) of outside background INTO the morph zone
            streak_len = int(rng.integers(2, 6))
            # Get source color from just outside the morph edge
            src_y = max(0, min(h_img-1, my0 + by - dy))
            src_x = max(0, min(w_img-1, mx0 + bx - dx_dir))
            src_color = bg[src_y, src_x].copy()
            for s in range(streak_len):
                py = by + dy * s
                px = bx + dx_dir * s
                if 0 <= py < mh and 0 <= px < mw and not morph_ink[py, px]:
                    # Blend original bg color into the morphed pixel
                    blend = rng.uniform(0.4, 0.85)
                    morph_region[py, px] = np.clip(
                        morph_region[py, px].astype(np.float32) * (1 - blend) + src_color.astype(np.float32) * blend,
                        0, 255
                    ).astype(np.uint8)



SYNTH_INK_TAMPER_REGISTRY = {
    'smooth_row_shift': _synth_smooth_row_shift,
    'copy_move': _synth_copy_move,
    'non_ink_background_noise': _synth_non_ink_background_noise,
    'faint_horizontal_lines': _synth_faint_horizontal_lines,
    'field_static_noise': _synth_field_static_noise,
    'clean_text_reprint': _synth_clean_text_reprint,
    'cool_blue_bg_filter': _synth_cool_blue_bg_filter,
    'diffusion_spread': _synth_diffusion_spread,
    'bg_color_morph': _synth_bg_color_morph,
}

def synth_render_parts(bg, spec, seed=0, tamper=False):
    rng = np.random.default_rng(seed)
    img = bg.copy()
    h, w = img.shape[:2]
    # Field-kind font bias: ID numbers lean monospace/OCR-B, long address/authority lean condensed.
    field_kind = spec.get('field_kind', '')
    if field_kind == 'id_number' and rng.random() < 0.55:
        font_prefer = 'mono'
    elif field_kind in ('address', 'authority') and (spec.get('is_long_value') or rng.random() < 0.5):
        font_prefer = 'condensed'
    else:
        font_prefer = None
    font = choose_synth_font(rng, size=int(rng.integers(18, 30)), text=' '.join([spec['final_label_text'], spec['final_value_text']]), prefer=font_prefer)
    scale = 1.0
    thickness = int(rng.integers(1, 3))
    color = int(rng.integers(25, 85))
    # Match PaddleOCR-field crops: main text is centered inside the field crop.
    # Compute vertical placement after width fitting so the rendered text block is centered too.
    if spec['field_style'] == 'value_only':
        label_part = ''
        joiner = ''
    elif spec['field_style'] == 'label_value_colon':
        label_part = spec['final_label_text']
        joiner = ': '
    elif spec['field_style'] == 'label_value_space':
        label_part = spec['final_label_text']
        joiner = '  '
    else:
        label_part = spec['final_label_text']
        joiner = ': '
        cv2.rectangle(img, (4, 8), (w - 5, h - 8), (int(color + 110), int(color + 110), int(color + 110)), 1)
    value_part = spec['final_value_text']
    value_segments = spec.get('value_segments', [('normal', value_part)])
    label_draw = label_part + joiner
    label_value_gap = int(rng.integers(2, 8))
    full_text_for_font = ' '.join([spec['final_label_text'], spec['final_value_text']])

    # Fit the whole main OCR text block into the synthetic field crop.
    # Never truncate: reduce font size until width and height both fit with padding.
    max_text_w = max(24, w - 24)
    max_text_h = max(12, h - 18)
    for fitted_size in range(int(font.size), 11, -1):
        candidate_font = choose_synth_font(rng, size=fitted_size, text=full_text_for_font, prefer=font_prefer)
        label_w = pil_text_size(label_draw, candidate_font)[0] if label_draw else 0
        value_w = sum(pil_text_size(text, candidate_font)[0] for _, text in value_segments)
        total_text_w = label_w + (label_value_gap if label_draw else 0) + value_w
        label_h = pil_text_size(label_draw, candidate_font)[1] if label_draw else 0
        value_h = max([pil_text_size(text, candidate_font)[1] for _, text in value_segments] or [1])
        text_h = max(label_h, value_h)
        font = candidate_font
        if total_text_w <= max_text_w and text_h <= max_text_h:
            break
    y = int(np.clip((h - text_h) / 2 + rng.integers(-1, 2), 6, max(6, h - text_h - 6)))
    align_mode = rng.choice(['left', 'center', 'right', 'random'], p=[0.02, 0.94, 0.02, 0.02])
    if align_mode == 'center':
        x = int(np.clip((w - total_text_w) / 2 + rng.integers(-5, 6), 8, max(8, w - total_text_w - 8)))
    elif align_mode == 'right':
        x = int(max(8, w - total_text_w - rng.integers(12, 30)))
    elif align_mode == 'random':
        x = int(rng.integers(8, max(9, w - total_text_w - 8)))
    else:
        x = int(rng.integers(14, 32))
    x = int(np.clip(x, 4, max(4, w - 8)))
    bboxes = {'align_mode': align_mode, 'horizontal_center_offset_px': float((x + total_text_w / 2) - (w / 2)), 'vertical_center_offset_px': float((y + text_h / 2) - (h / 2))}

    if rng.random() < 0.25:
        img = add_edge_text_decoys(img, rng, font, color, thickness)
    # Snapshot AFTER decoys, BEFORE main text — so ink mask captures only main text pixels.
    bg_with_decoys = img.copy()
    # ICAO 9303: second-language captions are commonly italic. Occasionally italicize the label only.
    label_font = font
    if label_draw and rng.random() < 0.15:
        label_font = choose_synth_font(rng, size=int(font.size), text=label_draw, prefer=font_prefer, italic_p=1.0)

    def put(text, pos, text_color=color, blur=False, text_thickness=thickness):
        # Use the fitted font for main text so measured widths match rendered widths.
        pil_put_text(img, text, pos, font, text_color, blur=blur)

    if label_draw:
        pil_put_text(img, label_draw, (x, y), label_font, color, blur=False)
        lw, lh = pil_text_size(label_draw, label_font)
        bboxes['label'] = (x, max(0, y - 3), min(w, x + lw), min(h, y + lh + 6))
        vx = x + lw + label_value_gap
    else:
        vx = x

    segment_boxes = []
    cursor_x = vx
    max_h = 1
    for tag, text in value_segments:
        sw, sh = pil_text_size(text, font)
        if text:
            segment_boxes.append({
                'tag': tag,
                'text': text,
                'x0': cursor_x,
                'y0': max(0, y - 3),
                'x1': min(w, cursor_x + sw + 2),
                'y1': min(h, y + sh + 6),
            })
        cursor_x += sw
        max_h = max(max_h, sh)
    value_box = (vx, max(0, y - 3), min(w, cursor_x + 2), min(h, y + max_h + 6))
    fake_boxes = [b for b in segment_boxes if b['tag'] == 'fake_part']
    if tamper and spec['target_span'] == 'value_part' and fake_boxes:
        tamper_box = (
            min(b['x0'] for b in fake_boxes), min(b['y0'] for b in fake_boxes),
            max(b['x1'] for b in fake_boxes), max(b['y1'] for b in fake_boxes),
        )
    else:
        tamper_box = value_box

    # --- Render all text cleanly first (no per-segment tamper during drawing) ---
    cursor_x = vx
    for tag, text in value_segments:
        put(text, (cursor_x, y))
        cursor_x += pil_text_size(text, font)[0]

    bboxes['value'] = value_box
    bboxes['tamper_target'] = tamper_box

    # --- Apply ink-aware tamper effect on the targeted text span only ---
    if tamper and spec['tamper_type'] in SYNTH_INK_TAMPER_REGISTRY:
        # Diff against bg_with_decoys (not original bg) so decoy pixels are excluded from ink mask.
        ink_mask = _synth_ink_mask(img, bg_with_decoys)
        inpainted_bg = _synth_clean_background(img, ink_mask)
        lbl_box = bboxes.get('label')
        union_box = value_box if not lbl_box else (
            min(lbl_box[0], value_box[0]), min(lbl_box[1], value_box[1]),
            max(lbl_box[2], value_box[2]), max(lbl_box[3], value_box[3]),
        )
        # Restrict the effect to the span the spec says is tampered, so e.g. a 'value'
        # tamper never touches the field label and a 'value_part' tamper hits only the
        # spliced characters. Background-only effects still self-mask to non-ink pixels.
        span = spec['target_span']
        if span == 'value_part':
            target_box = tamper_box            # fake-part sub-box (falls back to value_box)
        elif span == 'value':
            target_box = value_box
        elif span == 'label' and lbl_box:
            target_box = lbl_box
        else:                                  # 'both' / 'background' (or label missing) -> whole field
            target_box = union_box
        pad = 4 if span == 'value_part' else 6
        tx0 = max(0, int(target_box[0]) - pad)
        ty0 = max(0, int(target_box[1]) - pad)
        tx1 = min(w, int(target_box[2]) + pad)
        ty1 = min(h, int(target_box[3]) + pad)
        # Keep value/label spans on their own side so the pad never bleeds across the
        # label-value gap into the other span's pixels.
        if span in ('value', 'value_part') and lbl_box:
            tx0 = max(tx0, int(lbl_box[2]))    # don't reach left into the label
        elif span == 'label' and lbl_box:
            tx1 = min(tx1, int(value_box[0]))  # don't reach right into the value
        tamper_fn = SYNTH_INK_TAMPER_REGISTRY[spec['tamper_type']]
        tamper_fn(img, inpainted_bg, ink_mask, tx0, ty0, tx1, ty1, rng)
        bboxes['tamper_effect'] = spec['tamper_type']
        bboxes['tamper_span'] = span
        bboxes['tamper_region'] = (tx0, ty0, tx1, ty1)

    return img, bboxes

def synth_capture_degradation(img, rng):
    """Simulate photo/scan capture. Strength scales with image size —
    tiny images get lighter degradation to avoid becoming unreadable."""
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    # Scale factor: 1.0 for large images (200+px), 0.3 for tiny (30px)
    size_factor = float(np.clip((min(h, w) - 20) / 180, 0.2, 1.0))
    out *= rng.uniform(0.93, 1.07)                                   # exposure (lighter)
    out += rng.uniform(-6, 6)                                        # brightness
    out += rng.normal(0, 2, 3)                                       # white-balance tint
    if rng.random() < 0.6 * size_factor:                             # resolution loss — skip for tiny
        scale = rng.uniform(0.75, 0.95)  # never below 75% — text must stay readable
        small = cv2.resize(out.clip(0, 255).astype(np.uint8),
                           (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if rng.random() < 0.5 * size_factor:                             # optical blur — lighter
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=rng.uniform(0.2, max(0.3, 0.7 * size_factor)))
    out += rng.normal(0, rng.uniform(1.0, max(1.5, 3.5 * size_factor)), out.shape)  # sensor noise
    out = out.clip(0, 255).astype(np.uint8)
    q = int(rng.integers(60, 95))                                    # JPEG quality: higher min
    ok, enc = cv2.imencode('.jpg', cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out

def make_synth_dataset_sample(sample_id, label, seed=0, force_tamper_type=None):
    """Generate a synthetic field crop that matches real PaddleOCR extraction dimensions.

    Key change: text rendered FIRST at natural size, then canvas = tight bbox + realistic
    padding (matching what doc_field_crops extracts from PaddleOCR boxes: ~15px horizontal,
    ~5-8px vertical).  This eliminates the old "tiny text in huge canvas" problem.
    """
    spec = synth_text_spec(seed, label, force_tamper_type=force_tamper_type)
    rng = np.random.default_rng(seed)

    # --- Step 1: pick font and measure text at natural size ---
    full_text = spec['final_value_text']
    if spec['field_style'] != 'value_only':
        full_text = spec['final_label_text'] + ': ' + full_text
    font_size = int(rng.integers(18, 32))
    field_kind = spec.get('field_kind', '')
    if field_kind == 'id_number' and rng.random() < 0.55:
        font_prefer = 'mono'
    elif field_kind in ('address', 'authority') and (spec.get('is_long_value') or rng.random() < 0.5):
        font_prefer = 'condensed'
    else:
        font_prefer = None
    font = choose_synth_font(rng, font_size, text=full_text, prefer=font_prefer)
    tw, th = pil_text_size(full_text, font)

    # --- Step 2: create tight canvas with PaddleOCR-realistic padding ---
    pad_x = int(rng.integers(10, 22))      # matches doc_field_crops ±15px
    pad_y = int(rng.integers(8, 14))       # matches doc_field_crops ±10px
    canvas_w = tw + 2 * pad_x
    canvas_h = th + 2 * pad_y
    # Clamp to reasonable bounds (very short text still gets a minimum width)
    canvas_w = max(60, min(canvas_w, 900))
    canvas_h = max(24, min(canvas_h, 90))

    bg = synth_diverse_background(h=canvas_h, w=canvas_w, seed=seed + 1000)
    img, bboxes = synth_render_parts(bg, spec, seed=seed + 2000, tamper=bool(label))
    img = synth_capture_degradation(img, np.random.default_rng(seed + 3000))

    # --- Step 3: tight-crop to actual ink content (like PaddleOCR would) ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)
    ink = ((gray < local - 18) & (gray < 185)) | (gray < 90)
    ink = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
    if ink.any():
        ys, xs = np.where(ink)
        crop_pad_x = int(rng.integers(8, 18))
        crop_pad_y = int(rng.integers(8, 14))
        cx0 = max(0, int(xs.min()) - crop_pad_x)
        cy0 = max(0, int(ys.min()) - crop_pad_y)
        cx1 = min(img.shape[1], int(xs.max()) + 1 + crop_pad_x)
        cy1 = min(img.shape[0], int(ys.max()) + 1 + crop_pad_y)
        img = img[cy0:cy1, cx0:cx1]

    panel = forensic_panel_from_crop(img)
    meta = {
        'sample_id': sample_id,
        'label': int(label),
        'tamper_type': spec['tamper_type'],
        'field_kind': spec['field_kind'],
        'field_style': spec['field_style'],
        'is_long_value': spec['is_long_value'],
        'target_span': spec['target_span'],
        'fake_span_chars': spec['fake_span_chars'],
        'label_text': spec['label_text'],
        'value_text': spec['value_text'],
        'fake_value_text': spec['fake_value_text'],
        'final_text': spec['final_value_text'] if spec['field_style'] == 'value_only' else f"{spec['final_label_text']}: {spec['final_value_text']}",
        'raw_shape': img.shape,
        'panel_shape': panel.shape,
    }
    return img, panel, meta


# ── Apply tamper effects to real field crops (from train_dinov3_finetune.py) ──

def apply_synth_tamper_to_real(crop, rng, effect=None):
    """Apply a synthetic tamper effect to a real text field crop.

    Uses real_ink_mask from forensic_panels to detect ink, inpaints background,
    then applies a random effect from SYNTH_INK_TAMPER_REGISTRY.
    """
    from .forensic_panels import real_ink_mask
    out = crop.copy()
    ink = real_ink_mask(out)
    if ink.sum() < 15:
        return out, None
    mask_u8 = cv2.dilate(ink.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), 1)
    bg = cv2.inpaint(out, mask_u8, 3, cv2.INPAINT_TELEA)
    ys, xs = np.where(ink)
    h, w = ink.shape
    x0, y0 = max(0, int(xs.min()) - 4), max(0, int(ys.min()) - 3)
    x1, y1 = min(w, int(xs.max()) + 5), min(h, int(ys.max()) + 4)
    if effect is None:
        r = rng.random()
        if r < 0.25:
            effect = "clean_text_reprint"
        elif r < 0.40:
            effect = "smooth_row_shift"
        elif r < 0.48:
            effect = "cool_blue_bg_filter"
        elif r < 0.58:
            effect = "bg_color_morph"
        elif r < 0.75:
            effect = "diffusion_spread"
            mode = rng.choice(["both", "only_s", "only_t"])
            s = float(rng.uniform(0.1, 0.5)) if mode != "only_t" else 0.0
            t = float(rng.uniform(0.05, 0.5)) if mode != "only_s" else 0.0
            SYNTH_INK_TAMPER_REGISTRY[effect](out, bg, ink, x0, y0, x1, y1, rng, strength=s, thickness=t)
            return out, effect
        else:
            others = [k for k in SYNTH_INK_TAMPER_REGISTRY
                      if k not in ("clean_text_reprint", "smooth_row_shift", "cool_blue_bg_filter", "diffusion_spread")]
            effect = str(rng.choice(others))
    SYNTH_INK_TAMPER_REGISTRY[effect](out, bg, ink, x0, y0, x1, y1, rng)
    return out, effect