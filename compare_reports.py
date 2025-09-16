import json
import re
import os
import sys
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List


# Вопросы для точного разбиения
QUESTION_PATTERNS = [
    r"1[.)]\s*как вы её воспринимаете, оцениваете, переживаете и преодолеваете[^(]*",
    r"2[.)]\s*каковы ваши цели в этой ситуации",
    r"3[.)]\s*какие возможности и ограничения есть у вас при достижении цели",
    r"4[.)]\s*нужна ли вам в этой ситуации помощь \(поддержка\) окружающих людей",
    r"5[.)]\s*если вс[её] сложится очень плохо, то что это будет\??\s*\(максимальный неуспех\)",
    r"6[.)]\s*опишите, что для вас будет максимально успешным выходом, разрешением ситуации\.?"
]


def map_token_positions(a: List[str], b: List[str]) -> List[int]:
    """
    Для двух списков токенов a и b возвращает список длины len(a)+1,
    где каждый элемент i — это позиция-разделитель из a в списке разделителей b.
    """
    sm = SequenceMatcher(None, a, b)
    mapping = [0] * (len(a) + 1)
    curr_pos1 = 0

    for tag, i1_start, i1_end, i2_start, i2_end in sm.get_opcodes():
        if tag == 'equal':
            # соответствие токенов
            for k in range(i1_start, i1_end):
                mapping[k] = i2_start + (k - i1_start)
            mapping[i1_end] = i2_end
            curr_pos1 = i1_end

        elif tag in ('delete', 'replace'):
            # удаление или замена: все позиции из a → началу участка в b
            for k in range(i1_start, i1_end):
                mapping[k] = i2_start
            mapping[i1_end] = i2_start
            curr_pos1 = i1_end

        elif tag == 'insert':
            # вставка в b: позиция-разделитель в a остается указывает на конец вставки
            mapping[curr_pos1] = i2_end

    return mapping

def detect_text_mismatch(ref_words, test_words, threshold=0.8):
    """
    Определяет, есть ли значительное несоответствие между текстами.
    Возвращает True, если тексты слишком сильно отличаются.
    """
    return ref_words != test_words


def format_text_mismatch_error(situation_title, block_num, ref_words, test_words):
    """
    Форматирует сообщение об ошибке сопоставления текстов
    """
    block_names = ["Ситуация", "Вопрос 1", "Вопрос 2", "Вопрос 3", "Вопрос 4", "Вопрос 5", "Вопрос 6"]
    block_name = block_names[block_num] if block_num < len(block_names) else f"Блок {block_num}"
    
    sm = SequenceMatcher(None, ref_words, test_words)
    similarity = sm.ratio()

    first_diff = next((i for i, (a, b) in enumerate(zip(ref_words, test_words)) if a != b), min(len(ref_words), len(test_words)))
    #выводим 10 слов перед и после различия
    if first_diff is not None:
        start = max(0, first_diff - 5)
        end_ref = min(len(ref_words), first_diff + 10)
        end_test = min(len(test_words), first_diff + 10)
        ref_context = ' '.join(ref_words[start:end_ref])
        test_context = ' '.join(test_words[start:end_test])
        if start > 0:
            ref_context = '...' + ref_context
            test_context = '...' + test_context
        if end_ref < len(ref_words):
            ref_context += '...'
        if end_test < len(test_words):
            test_context += '...'
    else:
        ref_context = ' '.join(ref_words)
        test_context = ' '.join(test_words)
    
    return f"ОШИБКА СОПОСТАВЛЕНИЯ: Ситуация '{situation_title}', {block_name}. Различие: {1-similarity:.1%}\n" \
              f"  Эталон: '{ref_context}'\n" \
              f"  Тест:   '{test_context}'\n" \



def map_eng(text):
    """
    Заменяет кириллицу на латиницу в тексте, где написание букв одинаковое.
    """
    mapping = {k: v for k, v in zip(
        'АВСЕНКМОРТУХ',
        'ABCEHKMOPTYX')}
    return ''.join([mapping.get(c, c) for c in text])


def map_rus(text):
    """
    Заменяет латиницу на кириллицу в тексте, где написание букв одинаковое.
    """
    mapping = {k: v for k, v in zip(
        'ABCEHKMOPTYX',
        'АВСЕНКМОРТУХ')}
    return ''.join([mapping.get(c, c) for c in text])


def split_by_questions(text):
    """
    Разбивает текст на блоки: ситуация (до 1) и по каждому из 6 вопросов.
    Возвращает список из 7 строк: [situation, answer1, ..., answer6].
    """
    # Найдем позиции начала каждого вопроса
    positions = []
    for pat in QUESTION_PATTERNS:
        match = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            positions.append((match.start(), match.group(0)))
        else:
            raise ValueError(f"Не найдена метка вопроса: {pat}")
    # Сортируем по позиции
    positions.sort(key=lambda x: x[0])
    parts = []
    # Ситуация: от начала текста до первого вопроса
    first_pos = positions[0][0]
    parts.append(text[:first_pos].strip())
    # Ответы: между метками
    for i in range(len(positions)):
        start = positions[i][0]
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        parts.append(text[start:end].strip())
    if len(parts) != 7:
        raise RuntimeError("Ожидалось 7 частей (ситуация + 6 ответов)")
    return parts


word_pattern = r"(?<![\*A-Za-zА-Яа-яЁё0-9])[A-Za-zА-Яа-яЁё0-9]+"
code_pattern = r"\*[A-Za-z0-9А-ЯЁ\.]+"


def extract_segments(answer_text):
    """
    Извлекает из блока список сегментов с текстом и кодами.
    Возвращает list of dict: {'text', 'codes', 'span'}.
    """
    segments = []
    # Найти все вхождения [...]
    for match in re.finditer(r"\[([^\]]+)\]", answer_text):
        content = match.group(1).strip()
        # print(content)
        codes = re.findall(code_pattern, content)
        codes = sorted(set([map_eng(x) for x in codes]))
        if '*' in content:
            text_only = content.split('*', 1)[0].strip(" \t\n,;")
        else:
            text_only = content.strip(" \t\n,;")
        # Позиция: число слов перед началом '['
        prefix = answer_text[:match.end()]
        prefix = re.sub(code_pattern, "", prefix)
        # Слово: буквы и цифры
        word_count = len(re.findall(word_pattern, prefix))
        segments.append({'text': text_only, 'codes': codes, 'position': word_count})
        #print(f'found {text_only = }, {codes = }, {word_count = }')
    # Найти коды вне скобок
    all_codes = [map_eng(c) for c in re.findall(code_pattern, answer_text)]
    in_brackets = [c for seg in segments for c in seg['codes']]
    extra = sorted({c for c in all_codes if c not in in_brackets})
    if extra:
        # print(f'{all_codes = }, {extra = }, {in_brackets = }')
        # Все такие коды считаем на позиции конца текста
        prefix = re.sub(code_pattern, "", answer_text)
        words = re.findall(word_pattern, prefix)
        #print(f'{words = }')
        word_count = len(words)
        # print(f'{word_count = }')

        segments.append({'text': None, 'codes': extra, 'position': word_count+1})
        #print(f'found extra {extra = }, {word_count = }')
    return segments


def parse_markup(text):
    """
    Возвращает dict: номер (0–6) → list of segments.
    0 — ситуация, 1–6 — ответы.
    """
    parts = split_by_questions(text)
    parsed = {}
    for idx, block in enumerate(parts):
        parsed[idx] = extract_segments(block)
    return parsed, parts


def map_code_locations(parsed):
    """Карта code → list of tuples (question номер, position)."""
    code_map = [defaultdict(list) for _ in parsed]
    for qnum, segments in parsed.items():
        for seg in segments:
            for code in seg['codes']:
                code_map[qnum][code].append(seg['position'])
    return code_map


def add_dicts(d1, d2):
    """Складывает два словаря, возвращает новый словарь."""
    result = {}
    for k in d1.keys() | d2.keys():
        result[k] = d1.get(k, 0) + d2.get(k, 0)
    return result


def add_many_dicts(d1, *dicts):
    """Складывает два словаря списков, возвращает новый словарь списков."""
    result = d1
    for d in dicts:
        result = add_dicts(result, d)
    return result


def compare_positions(ref_map, test_map, ref_block, test_block):
    """Сравнивает позиции в двух списках. Возвращает статистику и информацию о несоответствии текстов."""
    ref_block_clean = re.sub(code_pattern, "", ref_block)
    test_block_clean = re.sub(code_pattern, "", test_block)
    ref_words = re.findall(word_pattern, ref_block_clean)
    test_words = re.findall(word_pattern, test_block_clean)
    
    # Проверяем несоответствие текстов
    text_mismatch = detect_text_mismatch(ref_words, test_words)
    
    pos_map = map_token_positions(ref_words, test_words)
    pos_map.append(len(test_words)+1)
    #print(f'{pos_map = }')

    # apply to all positions in ref_map
    ref_map = {code: [pos_map[pos] for pos in positions] for code, positions in ref_map.items()}

    all_ref = set(ref_map.keys())
    all_test = set(test_map.keys())
    for k, v in ref_map.items():
        if not v:
            raise ValueError(f"Код {k} не найден в эталонном наборе")
    for k, v in test_map.items():
        if not v:
            raise ValueError(f"Код {k} не найден в тестовом наборе")

    missing = [{'code': c, 'expected': ref_map[c], 'found': []} for c in sorted(all_ref - all_test)]
    extra = [{'code': c, 'found': test_map[c], 'expected': []} for c in sorted(all_test - all_ref)]
    misplaced = []
    correct = []
    duplicates = []
    for c in sorted(all_ref & all_test):
        ref_locs = set(ref_map[c])
        test_locs = set(test_map[c])
        eq = ref_locs & test_locs
        extra_ref = sorted(ref_locs - test_locs)
        extra_test = sorted(test_locs - ref_locs)
        if eq:
            correct.append({'code': c, 'expected': eq, 'found': eq})

        lm = min(len(extra_ref), len(extra_test))
        if lm:
            misplaced.append({'code': c, 'expected': extra_ref[:lm], 'found': extra_test[:lm]})

        if len(extra_ref) < len(extra_test):
            duplicates.append({'code': c, 'expected': [], 'found': extra_test[lm:]})
        elif len(extra_ref) > len(extra_test):
            missing.append({'code': c, 'expected': extra_ref[lm:], 'found': []})

    ncodes = len(all_ref | all_test)

    numcases_by_code = dict(add_many_dicts({d['code']: len(d['found']) for d in duplicates},
                                           {d['code']: len(d['expected']) for d in missing},
                                           {d['code']: len(d['expected']) for d in misplaced},
                                           {d['code']: len(d['found']) for d in correct},
                                           {d['code']: len(d['found']) for d in extra}))
    # print(numcases_by_code)
    # print(f'{correct = }')
    # print(f'{duplicates = }')
    # print(f'{missing = }')
    # print(f'{extra = }')
    # print(f'{misplaced = }')
    ntotal = sum(numcases_by_code.values())
    ntotal_ref = sum(len(v) for v in ref_map.values())
    ntotal_test = sum(len(v) for v in test_map.values())

    total_duplicates = sum(len(d['found']) for d in duplicates)
    total_missing = sum(len(d['expected']) for d in missing)
    total_misplaced = sum(len(d['expected']) for d in misplaced)
    total_extra = sum(len(d['found']) for d in extra)
    total_correct = sum(len(d['found']) for d in correct)

    w_duplicates = {d['code']: len(d['found']) / numcases_by_code[d['code']] for d in duplicates}
    w_missing = {d['code']: len(d['expected']) / numcases_by_code[d['code']] for d in missing}
    w_misplaced = {d['code']: len(d['expected']) / numcases_by_code[d['code']] for d in misplaced}
    w_extra = {d['code']: len(d['found']) / numcases_by_code[d['code']] for d in extra}
    w_correct = {d['code']: len(d['found']) / numcases_by_code[d['code']] for d in correct}

    total_w_duplicates = sum(w_duplicates.values())
    total_w_missing = sum(w_missing.values())
    total_w_misplaced = sum(w_misplaced.values())
    total_w_extra = sum(w_extra.values())
    total_w_correct = sum(w_correct.values())

    stats = {'missing': len(missing), 'extra': len(extra), 'misplaced': len(misplaced), 'correct': len(correct),
             'duplicates': len(duplicates),
             'total': len(missing) + len(extra) + len(misplaced) + len(correct) + len(duplicates),
             'total_ref': len(ref_map), 
             'total_test': len(test_map),}
    cstats = {'missing': total_missing, 'extra': total_extra, 'misplaced': total_misplaced, 'correct': total_correct,
              'duplicates': total_duplicates, 'total': ntotal,
              'total_ref': ntotal_ref, 'total_test': ntotal_test}
    wstats = {'missing': total_w_missing, 'extra': total_w_extra, 'misplaced': total_w_misplaced,
              'correct': total_w_correct, 'duplicates': total_w_duplicates, 'total': ncodes, 'total_ref': len(ref_map), 'total_test': len(test_map)}
    errors = {'missing': missing, 'extra': extra, 'misplaced': misplaced, 'correct': correct, 'duplicates': duplicates}
    
    # Возвращаем также информацию о несоответствии текстов
    text_mismatch_info = None
    if text_mismatch:
        text_mismatch_info = {
            'ref_words': ref_words,
            'test_words': test_words
        }
    
    return stats, cstats, wstats, errors, text_mismatch_info


def print_stats(stats, title="Statistics", format="10", normalize='total'):
    print(f"{title}")
    print(f"   Missing:    {stats['missing']:{format}} ({stats['missing'] / stats[normalize]:6.2%})")
    print(f"   Extra:      {stats['extra']:{format}} ({stats['extra'] / stats[normalize]:6.2%})")
    print(f"   Duplicates: {stats['duplicates']:{format}} ({stats['duplicates'] / stats[normalize]:6.2%})")
    print(f"   Misplaced:  {stats['misplaced']:{format}} ({stats['misplaced'] / stats[normalize]:6.2%})")
    print(f"   Correct:    {stats['correct']:{format}} ({stats['correct'] / stats[normalize]:6.2%})")
    print(f"   {normalize[0].upper()}{normalize[1:]}:      {stats[normalize]:{format}}\n")


def compare_markups(ref_text, test_text, ignore_text_errors=False, situation_title="", ignore_codes=None):
    stats = {}
    cstats = {}
    wstats = {}
    errors = []
    text_mismatch_errors = []
    
    ref, ref_blocks = parse_markup(ref_text)
    test, test_blocks = parse_markup(test_text)
    ref_map = map_code_locations(ref)
    test_map = map_code_locations(test)
    
    # Фильтрация игнорируемых кодов (как будто их нет ни в эталоне, ни в тесте)
    if ignore_codes:
        ref_map = [{k: v for k, v in m.items() if k not in ignore_codes} for m in ref_map]
        test_map = [{k: v for k, v in m.items() if k not in ignore_codes} for m in test_map]
    
    for block_num, (rr, tt, rb, tb) in enumerate(zip(ref_map, test_map, ref_blocks, test_blocks)):
        stats_, cstats_, wstats_, errors_, text_mismatch_info = compare_positions(rr, tt, rb, tb)
        stats = add_dicts(stats, stats_)
        cstats = add_dicts(cstats, cstats_)
        wstats = add_dicts(wstats, wstats_)
        errors.append(errors_)
        
        # Сохраняем информацию об ошибках сопоставления текстов
        if text_mismatch_info and not ignore_text_errors:
            error_msg = format_text_mismatch_error(
                situation_title, 
                block_num, 
                text_mismatch_info['ref_words'], 
                text_mismatch_info['test_words']
            )
            text_mismatch_errors.append(error_msg)

    return stats, cstats, wstats, errors, text_mismatch_errors


codes_description = [
    ["А", "ЭМОЦИИ"],
    ["A1", "Эмоции: Позитивные интенсивные"],
    ["A2", "Эмоции: Позитивные неинтенсивные"],
    ["A3", "Эмоции: Негативные интенсивные"],
    ["A4", "Эмоции: Негативные неинтенсивные"],
    ["A5", "Эмоции: Спокойствие"],
    ["A7", "Эмоции: Другое"],
    ["Б", "ВРЕМЯ"],
    ["Б1", "Время: Упоминается"],
    ["Б2", "Время: Не упоминается"],
    ["Б3", "Время: Несрочная ситуация"],
    ["Б4", "Время: Вре́менная, проходящая ситуация"],
    ["Б5", "Время: Другое"],
    ["Б6", "Часто / длительно происходящая ситуация"],
    ["Е", "ЭНЕРГИЯ"],
    ["E2", "Низкий уровень энергии"],
    ["E3", "Высокий уровень энергии"],
    ["E4", "Энергия растет"],
    ["E5", "Энергия снижается"],
    ["E6", "Необходимость затрат энергии, приложения усилий"],
    ["E7", "Другое (энергия)"],
    ["M1", "Отсутствие мотивации"],
    ["С", "СТЕПЕНЬ И СУТЬ ТРУДНОСТИ"],
    ["C1", "Степень трудности: ситуация очень трудная"],
    ["C2", "Степень трудности: ситуация нетрудная"],
    ["C3", "Суть: не справиться вообще"],
    ["C4", "Суметь сделать всё"],
    ["C5", "Нужно успеть"],
    ["C6", "Нужно достичь максимального результата"],
    ["C7", "Трудности: Сомнения"],
    ["C8", "Трудности: Справиться со своим состоянием"],
    ["C9", "Другое (трудность)"],
    ["Ф1", "Содержание ситуации: Материально-бытовая сфера"],
    ["Ф1.1", "Материальные трудности"],
    ["Ф1.2", "Жилищные условия"],
    ["Ф2", "Содержание ситуации: Профессиональная сфера"],
    ["Ф2.1", "Профессиональная деятельность"],
    ["Ф2.4", "Отношения на работе, с коллегами, начальником или с преподавателем в вузе"],
    ["Ф2.2", "Учебная деятельность"],
    ["Ф3", "Содержание ситуации: Межличностная сфера"],
    ["Ф3.1", "Отношения с родителями"],
    ["Ф3.2", "Отношения с противоположным полом / с (однополым) партнером"],
    ["Ф3.3", "Отношения с детьми"],
    ["Ф3.4", "Отношения в семье"],
    ["Ф3.5", "Отношения с друзьями"],
    ["Ф3.6", "Общие ситуации межличностных отношений (кто конкретно – не указано)"],
    ["Ф3.7", "Одиночество"],
    ["Ф3.8", "Трудная ситуация происходит_у близких людей"],
    ["Ф4", "Содержание ситуации: Внутриличностная сфера"],
    ["Ф4.1", "Внутриличностные конфликты и трудности принятия решения"],
    ["Ф4.2", "Адаптация к новым условиям"],
    ["Ф4.3", "Волевая сфера"],
    ["Ф4.4", "Эмоциональная сфера"],
    ["Ф4.5", "Коммуникативная сфера"],
    ["Ф4.6", "Распределение времени"],
    ["Ф4.7", "Будущее"],
    ["Ф4.8", "Жизненный путь в целом"],
    ["Ф5", "Содержание ситуации: Общественная (макро) сфера"],
    ["Ф6", "Угроза жизни и здоровью"],
    ["Ф6.1", "Болезнь (своя)"],
    ["Ф6.2", "Болезни других (значимых) людей"],
    ["Ф6.3", "Смерть"],
    ["Ф6.4", "Экстремальные ситуации"],
    ["Ф6.5", "Изнасилование"],
    ["Ф7", "Другое (содержание ситуации)"],
    ["Ф№.№.П", "Подробности о содержании ситуации"],
    ["Ч2", "ТЖС касается двух сфер жизни"],
    ["Ч3", "ТЖС касается трех и более сфер ж."],
    ["Ч4", "Каких именно сфер"],
    ["1А", "ВАЛЕНТНОСТЬ ОЦЕНКИ"],
    ["1A2", "Оценка ситуации: Положительная"],
    ["1A3", "Оценка ситуации: Отрицательная"],
    ["1A4", "Оценка ситуации: Нейтральная"],
    ["1A5", "Оценка ситуации: Амбивалентная"],
    ["1В", "ОСНОВАНИЯ ОЦЕНКИ"],
    ["1B2", "Значимость"],
    ["1B3", "Влиятельность"],
    ["1B4", "Масштабность"],
    ["1B5", "Аргументы оценки"],
    ["1B7", "Вызов"],
    ["1B9", "Необходимость"],
    ["1B8", "Другое (основания оценки)"],
    ["1К", "КРИТЕРИИ ОЦЕНКИ"],
    ["1K1", "Контроль над ситуацией"],
    ["1K2", "Неподконтрольность ситуации"],
    ["1K3", "Непонимание ситуации"],
    ["1K4", "Высокая динамика ситуации"],
    ["1K5", "Трудность прогноза"],
    ["1K6", "Дилемма"],
    ["1K7", "Значимость для будущего"],
    ["1K9", "Оценка как препятствия"],
    ["1K8", "Другое (критерии оценки)"],
    ["1D", "КОПИНГ"],
    ["1D1", "Не указано"],
    ["1D2", "Невозможность преодоления"],
    ["1D4", "Планомерный копинг"],
    ["1D5", "Борьба"],
    ["1D6", "Положительная переоценка"],
    ["1D7", "Анализ или осознание опыта"],
    ["1D8", "Подбадривать себя"],
    ["1D9", "Самообвинение"],
    ["1D10", "Надежда на случай, судьбу, обстоятельства"],
    ["1D11", "Откладывание"],
    ["1D12", "Уход, отвлечение"],
    ["1D13", "Дистанцирование"],
    ["1D14", "Надежда на Бога"],
    ["1D15", "Указание на помощь других людей"],
    ["1D17", "Оценка действенности копинга"],
    ["1D16", "Другое (копинг)"],
    ["2A", "ЦЕЛЬ"],
    ["2A1", "Нет информации о цели"],
    ["2A2", "Приближение («к»)"],
    ["2A3", "Избегание («от»)"],
    ["2A4", "Сохранение имеющегося"],
    ["2A5", "Развитие, увеличение"],
    ["2A6", "Вернуть, как было раньше"],
    ["2A7", "Другое (цель)"],
    ["2B", "УРОВЕНЬ ЦЕЛИ"],
    ["2B2", "Минимальный уровень цели"],
    ["2B3", "Максимальный уровень цели"],
    ["3A", "ВОЗМОЖНОСТИ"],
    ["3A1", "Не упоминаются"],
    ["3A2", "Отмечается, что возможностей много"],
    ["3A3", "Отмечается, что возможности не обнаружены или отсутствуют"],
    ["3A5", "Необходимость хоть какой-то активности"],
    ["3A6", "Что-то не делать, прекратить, уменьшить"],
    ["3A7", "Собственные возможности"],
    ["3A8", "Саморазвитие"],
    ["3A9", "Социальный ресурс"],
    ["3A10", "Интернет-ресурсы и информационные технологии; информация вообще, литература"],
    ["3A11", "Материальные возможности"],
    ["3A12", "Другое (возможности)"],
    ["3B", "ОГРАНИЧЕНИЯ"],
    ["3B1", "Не упоминаются"],
    ["3B2", "Отмечается отсутствие ограничений"],
    ["3B4", "Ограничения графика работы / учебы"],
    ["3B5", "Множество дел и невозможность отказа от множественной задачи"],
    ["3B6", "Собственные ограничения"],
    ["3B7", "Социальное окружение"],
    ["3B8", "Информация, интернет-ресурсы"],
    ["3B9", "Материальные ограничения"],
    ["3B10", "Другое (ограничения)"],
    ["4A", "НЕОБХОДИМОСТЬ ПОМОЩИ"],
    ["4A1", "Нет необходимости"],
    ["4A2", "Сомнения в её возможности"],
    ["4A3", "Уверенность в её необходимости"],
    ["4A4", "Нужна при определенных условиях"],
    ["4A5", "Необязательность помощи"],
    ["4A6", "Другое (необходимость помощи)"],
    ["4B", "СОДЕРЖАНИЕ ПОМОЩИ"],
    ["4B1", "Нет информации"],
    ["4B2", "Эмоциональная поддержка"],
    ["4B3", "Информационная поддержка"],
    ["4B4", "Взаимодействие"],
    ["4B5", "Инструментальная поддержка"],
    ["4B6", "Перекладывание задачи на других"],
    ["4B7", "Поддержка (вообще)"],
    ["4B8", "Другое (содержание помощи)"],
    ["4C", "Указание на причины отказа от помощи"],
    ["4D", "Другое (помощь)"],
    ["5A", "ПРОГНОЗ НА МАКСИМАЛЬНЫЙ НЕУСПЕХ"],
    ["5A1", "Отсутствует"],
    ["5A3", "Невозможность неуспеха"],
    ["5B", "СОДЕРЖАНИЕ НЕУСПЕХА"],
    ["5B1", "Предельный неуспех"],
    ["5B2", "Утрата"],
    ["5B3", "Неполучение"],
    ["5B4", "Перенос результата на более поздний срок"],
    ["5B5", "Останется также плохо"],
    ["5B6", "Неудача, провал"],
    ["5B8", "Ухудшится"],
    ["5B7", "Другое"],
    ["5C", "ДРУГОЕ (неуспех)"],
    ["6A", "ПРОГНОЗ НА МАКСИМАЛЬНЫЙ УСПЕХ"],
    ["6A1", "Отсутствует"],
    ["6A2", "Определенный негативный"],
    ["6A4", "Неопределенный"],
    ["6B", "СОДЕРЖАНИЕ УСПЕХА"],
    ["6B1", "Появление чего-то нового"],
    ["6B2", "Избавление от чего-то"],
    ["6B3", "Вернуть утерянное («как было»)"],
    ["6B4", "Поддержание существующего положения"],
    ["6B5", "Фантазийный успех"],
    ["6B6", "Нереалистичный успех"],
    ["6B8", "Завершить"],
    ["6B7", "Другое"],
    ["6C", "ДРУГОЕ (успех)"]
]


def read_categories():
    #with open('categories.json', 'r', encoding='utf-8') as f:
    #    data = json.load(f)
    data = codes_description
    categories = defaultdict(lambda: "Unknown code")
    for cat, d, *_ in data:
        categories['*' + map_eng(cat)] = d
    return categories


def fix_codes_syntax(text):
    codes = [x[0] for x in codes_description if len(x[0]) > 1]
    codes = [map_eng(c) for c in codes] + [map_rus(c) for c in codes]
    escaped_tokens = [re.escape(c) for c in codes]
    pattern = (
        r'(?<!(?:\*|[A-Za-z0-9А-Яа-яеЁ\.]))('
        + '|'.join(escaped_tokens)
        + r')(?![A-Za-z0-9А-Яа-яеЁ\.])'
    )
    return re.sub(pattern, r'*\1', text)


categories_descr = read_categories()


def insert_substrings(x: str, inserts: dict) -> str:
    result = []
    for i in range(len(x) + 1):
        # Добавляем все вставки перед позицией i, если есть
        if i in inserts:
            result.extend(inserts[i])
        if i < len(x):
            result.append(x[i])

    return ''.join(result)


def generate_html(test_text, stats, cstats, wstats, errors, normalize='total', efficiency_config=None):
    """
    test_text — ваш тестовый текст целиком.
    stats — dict со «случаями» ошибок, как возвращает compare_markups.
    cstats — dict со «вхождениями» ошибок.
    errors — список из 7 элементов (для q0–q6),
             каждый элемент содержит dict с keys: missing, extra, misplaced, duplicates, correct.
    """
    efficiency_config = efficiency_config or defaultdict(lambda: 1.0)
    # 1) Цвета
    COLORS = {
        'correct': '#88ff88',  # светло-зелёный
        'extra': '#ff8888',  # светло-красный
        'misplaced': '#ffff88',  # светло-жёлтый
        'misplaced_expected': '#ff88ff',  # светло-розовый
        'duplicates': '#ffc080',  # светло-оранжевый
        'missing': '#88ffff'  # светло-бирюзовый
    }
    categories = categories_descr

    # Добавим «_entries» в stats, взяв их из cstats
    for k in cstats:
        stats[k + '_entries'] = cstats.get(k, 0)

    # Формируем HTML
    html_parts = [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Отчёт о сравнении разметки</title>',
        '<style>'
    ]

    for cls, color in COLORS.items():
        html_parts.append(f'.{cls} {{ background: {color}; }}')

    html_parts.extend([
        'body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }',
        'table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }',
        'th, td { border: 1px solid #999; padding: .5em .6em; }',
        'th { background-color: #f2f2f2; }',
        'h2 { color: #333; border-bottom: 1px solid #ddd; padding-bottom: 10px; }',
        'pre { white-space: pre-wrap; font-size: 1.1em; max-width: 900px; ',
        '      background-color: #f9f9f9; padding: 15px; border-radius: 5px; overflow-x: auto; }',
        '.text-block { margin-bottom: 30px; font-size: 1.1em; }',
        '.question-text { font-weight: bold; display: block; margin-top: 10px; margin-bottom: 5px; font-size: 1.1em; }',
        '.legend { display: flex; flex-wrap: wrap; margin: 20px 0; }',
        '.legend-item { display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px; }',
        '.legend-color { width: 20px; height: 20px; margin-right: 5px; border: 1px solid #ccc; }',
        '</style></head><body>',
        '<h1>Отчёт о сравнении разметки</h1>'
    ])

    # Добавим легенду
    html_parts.append('<div class="legend">')
    for kind, color in COLORS.items():
        html_parts.append(
            f'<div class="legend-item"><div class="legend-color" style="background:{color}"></div><span>{kind}</span></div>')
    html_parts.append('</div>')

    # Сводная таблица
    html_parts.append('<h2>Сводная статистика</h2><table><tr>'
                      '<th>Тип</th>'+ (
                      '<th>Случаев</th>'
                      '<th>Случаев (%)</th>' if normalize == 'total' else '') +
                      '<th>Вхождений</th>'
                      '<th>Вхождений (%)</th>'
                      '<th>Взвешенных вхождений</th>'
                      '<th>Взвешенных вхождений (%)</th>'
                      '</tr>'
                      )
    ru_kind = {
        'correct': 'Верные',
        'extra': 'Лишние',
        'misplaced': 'Смещённые',
        'duplicates': 'Дубликаты',
        'missing': 'Пропущенные',
        normalize: 'Всего'
    }

    # Безопасное деление для процентов
    def safe_div(a, b):
        return a / b if b else 0

    for kind in ['correct', 'misplaced', 'duplicates', 'extra', 'missing', normalize]:
        extra = (kind == 'extra' or kind == 'duplicates') and normalize == 'total_ref'
        plus = '+' if extra else ''
        color = ' style=\"color:#C00\"' if extra and normalize == 'total_ref' else ''
        html_parts.append(
            f"<tr><td>{ru_kind[kind]}</td>"
            + (
                f"<td align='right'>{stats.get(kind, 0)}</td>"
                f"<td align='right'{color}>{safe_div(stats.get(kind, 0), stats.get(normalize, 0)):.1%}</td>"
                if normalize == 'total' else ''
            )
            + f"<td align='right'>{cstats.get(kind, 0)}</td>"
            + f"<td align='right'{color}>{plus}{safe_div(cstats.get(kind, 0), cstats.get(normalize, 0)):.1%}</td>"
            + f"<td align='right'>{wstats.get(kind, 0):.2f}</td>"
            + f"<td align='right'{color}>{plus}{safe_div(wstats.get(kind, 0), wstats.get(normalize, 0)):.1%}</td>"
            + f"</tr>"
        )
    html_parts.append('</table>')
    # 2) Разбить тестовый текст на 7 блоков
    try:
        parts = split_by_questions(test_text)
    except ValueError as e:
        print(f"Test:\n-------------------\n {test_text}\n--------------------")
        raise
    

    # 3) Для каждого блока подготовить карту вставок и обёрток
    for qnum, block in enumerate(parts):
        # Для первого блока (ситуации) просто показываем содержимое
        match, match_text = None, ""
        if qnum == 0:
            html_parts.append('<div class="question-text">Ситуация</div>')
        else:
            # Выделяем текст вопроса жирным, но не увеличиваем его размер
            match = re.search(QUESTION_PATTERNS[qnum-1], block, flags=re.IGNORECASE | re.MULTILINE)
            match_text = block.split('\n')[0] if match else f"Вопрос {qnum}"
            html_parts.append(f'<div class="question-text">{match_text}</div>')

        html_parts.append("<div class='text-block'>")

        errs = errors[qnum]
        # карта: position → list of (code, class)
        insert_map = {}
        wrap_map = {}
        
        # вспомогалка
        def add_to(m, pos, code, cls):
            m.setdefault(pos, []).append((code, cls))

        # missing: вставка бирюзой
        for e in errs['missing']:
            for pos in e['expected']:
                add_to(wrap_map, pos, e['code'], 'missing')

        # misplaced:
        #  - найденные — жёлтым
        #  - ожидаемые — розовым
        for e in errs['misplaced']:
            for pos in e['found']:
                add_to(wrap_map, pos, e['code'], 'misplaced')
            for pos in e['expected']:
                add_to(wrap_map, pos, e['code'], 'misplaced_expected')

        # extra: найденные красным
        for e in errs['extra']:
            for pos in e['found']:
                add_to(wrap_map, pos, e['code'], 'extra')

        # duplicates: найденные оранжевым
        for e in errs['duplicates']:
            for pos in e['found']:
                add_to(wrap_map, pos, e['code'], 'duplicates')

        # correct: найденные зелёным
        for e in errs['correct']:
            for pos in e['found']:
                add_to(wrap_map, pos, e['code'], 'correct')

        cls_name_ru = {
            'correct': 'верно',
            'extra': 'лишнее',
            'misplaced': 'смещено',
            'misplaced_expected': 'смещено (должно быть здесь)',
            'duplicates': 'дубликат',
            'missing': 'пропущено'
        }

        # 4) Пройти по блоку «слово за словом»
        def wrap_code(code, cls):
            color = COLORS[cls] if cls else 'transparent'
            return f"<span style='background:{color}' title=\"{cls_name_ru[cls]}: {categories.get(code, 'unknown code')}\">{code}</span>"

        block = re.sub(r"(?:\*[A-Za-z0-9А-ЯЁ\.]+\s*[;,]\s*)*\*[A-Za-z0-9А-ЯЁ\.]+\s*", "", block)

        words_pos = re.finditer(r"(?<![\*A-Za-zА-Яа-яЁё0-9])[A-Za-zА-Яа-яЁё0-9]+[.?)]*", block)
        words = re.findall(r"(?<![\*A-Za-zА-Яа-яЁё0-9])[A-Za-zА-Яа-яЁё0-9]+[.?)]*", block)
        #print(f'{words = }')
        word_end_map = [0] + [m.end() for m in words_pos] + [len(block)]
        if any(pos >= len(word_end_map) for pos, codes in wrap_map.items()):
            print(f'errs = {errs}\n-----\nwords = {words}\nlen(words) = {len(words)}\nwrap_map = {wrap_map}')
            raise ValueError("Позиция слова выходит за пределы текста")
        insert_map = {word_end_map[pos]: " " + ", ".join([wrap_code(code, cls) for code, cls in codes]) for pos, codes
                      in wrap_map.items()}
        text = insert_substrings(block, insert_map)

        if match:
            text = text.replace(match_text, "", 1).strip()
        text = text.replace(" ]", "] ").replace("[ ", " [")
        html_parts.append(f"<pre>{text}</pre>")
        html_parts.append("</div>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def test():
    cat_desc = categories_descr
    ref_text = open('reference.txt', encoding='utf-8').read()
    test_text = open('test.txt', encoding='utf-8').read()
    stats, cstats, wstats, errors, text_mismatch_errors = compare_markups(ref_text, test_text, ignore_text_errors=True)
    print_stats(stats, "Статистика случаев")
    print_stats(cstats, "Статистика вхождений")
    print_stats(wstats, "Взвешенная статистика вхождений", "10.2f")

    print("Missing: ")
    for i, ee in enumerate(errors):
        for e in ee['missing']:
            print(f'   {i}: ', e['code'], e['expected'], "\t// ", cat_desc[e['code']])
    print("Extra: ")
    for i, ee in enumerate(errors):
        for e in ee['extra']:
            print(f'   {i}: ', e['code'], e['found'], "\t// ", cat_desc[e['code']])
    print("Misplaced: ")
    for i, ee in enumerate(errors):
        for e in ee['misplaced']:
            print(f'   {i}: ', e['code'], e['expected'], '--->', e['found'], "\t// ", cat_desc[e['code']])
    print("Duplicates: ")
    for i, ee in enumerate(errors):
        for e in ee['duplicates']:
            print(f'   {i}: ', e['code'], e['found'], "\t// ", cat_desc[e['code']])


def generate_multi_situation_report(situations_dict, reference_dict, normalize='total', efficiency_config=None, ignore_text_errors=False, ignore_codes=None):
    """
    Создает HTML-отчет для нескольких ситуаций с общей сводкой и детализацией

    :param situations_dict: словарь {название: разметка_ситуации}
    :param reference_dict: словарь {название: эталонная_разметка}
    :param ignore_text_errors: не выводить ошибки сопоставления текстов
    :return: строка -- HTML-код отчета
    """
    efficiency_config = efficiency_config or {}
    # Собираем статистику по всем ситуациям
    situation_results = {}
    all_stats = {}
    all_cstats = {}
    all_wstats = {}
    all_text_errors = []

    not_in_reference = []
    for title, test_text in situations_dict.items():
        if title not in reference_dict:
            print(f"Предупреждение: '{title}' не найден в эталонных данных.")
            not_in_reference.append(title)
            continue
        # Сравниваем с эталонной разметкой
        reference_text = reference_dict.get(title)
        try:
            stats, cstats, wstats, errors, text_mismatch_errors = compare_markups(
                reference_text, test_text, ignore_text_errors, title, ignore_codes
            )
        except Exception as e:
            print(f"Ошибка при сравнении '{title}': {e}")
            continue
        
        # Выводим ошибки сопоставления текстов на экран
        if text_mismatch_errors and not ignore_text_errors:
            for error_msg in text_mismatch_errors:
                print(error_msg)
            all_text_errors.extend(text_mismatch_errors)
        
        situation_results[title] = {
            'stats': stats,
            'cstats': cstats,
            'wstats': wstats,
            'errors': errors,
            'text': test_text
        }

        # Объединяем статистику
        all_stats = add_dicts(all_stats, stats) if all_stats else stats.copy()
        all_cstats = add_dicts(all_cstats, cstats) if all_cstats else cstats.copy()
        all_wstats = add_dicts(all_wstats, wstats) if all_wstats else wstats.copy()

    # Выводим итоговое сообщение об ошибках сопоставления
    if all_text_errors and not ignore_text_errors:
        print(f"\n=== ИТОГО НАЙДЕНО {len(all_text_errors)} ОШИБОК СОПОСТАВЛЕНИЯ ТЕКСТОВ ===\n")

    if not situation_results:
        print("Нет ситуаций для анализа.")
        return ""

    # Формируем HTML
    html_parts = [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8">',
        '<title>Сводный отчёт по разметке ситуаций</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 0; padding: 0; }',
        '.page-container { display: flex; min-height: 100vh; }',
        '.sidebar { width: 250px; background-color: #f5f5f5; border-right: 1px solid #ddd; padding: 20px; ',
        '          position: fixed; left: 0; top: 0; bottom: 0; overflow-y: auto; }',
        '.main-content { flex: 1; padding: 20px; margin-left: 290px; }' # margin-left: 290px; max-width:1000px;}'
        '.content-container { max-width: 1000px; margin: 0 auto; padding: 20px; }'
        'table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }',
        'th, td { border: 1px solid #999; padding: .5em .6em; }',
        'th { background-color: #f2f2f2; }',
        'h1 { margin-top: 0; padding-bottom: 15px; border-bottom: 1px solid #ddd; }',
        'h2 { color: #333; border-bottom: 1px solid #ddd; padding-bottom: 10px; }',
        'pre { white-space: pre-wrap; font-size: 1.1em; background-color: #f9f9f9; padding: 15px; border-radius: 5px; overflow-x: auto; }',
        '.text-block { margin-bottom: 30px; font-size: 1.1em; }',
        '.question-text { font-weight: bold; display: block; margin-top: 10px; margin-bottom: 5px; font-size: 1.1em; }',
        '.situation-card { border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; }',
        '.situation-header { background-color: #f5f5f5; padding: 10px 15px; cursor: pointer; }',
        '.situation-content { padding: 0 15px; max-height: 0; overflow: hidden; transition: max-height 0.3s ease; }',
        '.nav-title { font-weight: bold; font-size: 1.2em; margin-bottom: 15px; border-bottom: 1px solid #ccc; padding-bottom: 8px; }',
        '.nav-links { display: flex; flex-direction: column; }',
        '.nav-links a { text-decoration: none; padding: 6px 10px; color: #333; }',
        '.nav-links a:hover { background: #e0e0e0; }',
        '.legend { display: flex; flex-wrap: wrap; margin: 20px 0; }',
        '.legend-item { display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px; }',
        '.legend-color { width: 20px; height: 20px; margin-right: 5px; border: 1px solid #ccc; }',
        '.tab-buttons { display: flex; gap: 10px; margin-bottom: 15px; }',
        '.tab-button { padding: 8px 16px; cursor: pointer; background: #f0f0f0; border: 1px solid #ddd; border-radius: 4px; }',
        '.tab-button.active { background: #ddd; font-weight: bold; }',
        '.tab-content { display: none; }',
        '.tab-content.active { display: block; }',
        '</style>',
        '<script>',
        '/* JavaScript без изменений */',
        'function toggleSituation(id, forceState = null) {',
        '  const content = document.getElementById("content-" + id);',
        '  if (forceState === true || (forceState === null && !content.style.maxHeight)) {',
        '    content.style.maxHeight = content.scrollHeight + "px"; ',
        '  } else if (forceState === false || (forceState === null && content.style.maxHeight)) {',
        '    content.style.maxHeight = null;',
        '  }',
        '}',
        'function switchTab(tabType) {',
        '  document.querySelectorAll(".tab-button").forEach(btn => {',
        '    btn.classList.remove("active");',
        '  });',
        '  document.getElementById("btn-" + tabType).classList.add("active");',
        '  document.querySelectorAll(".tab-content").forEach(content => {',
        '    content.classList.remove("active");',
        '  });',
        '  document.getElementById("tab-" + tabType).classList.add("active");',
        '  document.getElementById("tab-compare-" + tabType).classList.add("active");',
        '}',
        'function openSituationFromHash() {',
        '  const hash = window.location.hash;',
        '  if (hash && hash.startsWith("#situation-")) {',
        '    const situationElement = document.querySelector(hash);',
        '    if (situationElement) {',
        '      // Находим ID в data-атрибуте или через порядковый номер',
        '      const allSituations = Array.from(document.querySelectorAll(".situation-card"));',
        '      const index = allSituations.indexOf(situationElement);',
        '      if (index >= 0) {',
        '        // Раскрываем содержимое',
        '        toggleSituation(index, true);',
        '        // Прокрутка с учетом шапки',
        '        setTimeout(() => {',
        '          window.scrollTo(0, situationElement.offsetTop - 20);',
        '        }, 100);',
        '      }',
        '    }',
        '  }',
        '}',

        '// Запускаем при загрузке страницы',
        'document.addEventListener("DOMContentLoaded", openSituationFromHash);',

        '// Запускаем при изменении хэша (переход по ссылкам)',
        'window.addEventListener("hashchange", openSituationFromHash);',
        '</script>',
        '</head><body>',
        '<div class="page-container">',
        '<div class="sidebar">',
        '<div class="nav-title">Оглавление</div>',
        '<div class="nav-links">',
        '<a href="#summary">Общая сводка</a>'
    ]

    make_safe_id = lambda title: re.sub(r'[^a-zA-Z0-9_-]', '_', title)

    # Добавляем ссылки на ситуации в боковую навигацию
    for title in situation_results.keys():
        # Создаем безопасный ID для ситуации
        safe_id = make_safe_id(title)
        html_parts.append(f'<a href="#situation-{safe_id}">{title}</a>')

    html_parts.append('</div></div>')  # закрываем навигацию и боковую панель

    # Открываем основной контент
    html_parts.append('<div class="main-content"><div class="content-container">')
    html_parts.append('<h1>Сводный отчёт по разметке ситуаций</h1>')

    # Добавим легенду с цветами
    COLORS = {
        'верные': '#88ff88',      # светло-зелёный
        'лишние': '#ff8888',      # светло-красный
        'смещённые': '#ffff88',   # светло-жёлтый
        'дубликаты': '#ffc080',   # светло-оранжевый
        'пропущенные': '#88ffff'  # светло-бирюзовый
    }

    # Общая сводная таблица с переключателями
    html_parts.extend([
        '<section id="summary">',
        '<h2>Общая сводка</h2>',
        '<div class="tab-buttons">'] + ([
        '<button id="btn-cases" class="tab-button" onclick="switchTab(\'cases\')">Случаи</button>'] if normalize == 'total' else []) + [
        '<button id="btn-entries" class="tab-button active" onclick="switchTab(\'entries\')">Вхождения</button>',
        '<button id="btn-weighted" class="tab-button" onclick="switchTab(\'weighted\')">Взвешенные вхождения</button>',
        '</div>'
    ])

    # Сводные таблицы для каждого типа статистики
    stat_types = {
        'cases': ('Случаи', all_stats),
        'entries': ('Вхождения', all_cstats),
        'weighted': ('Взвешенные вхождения', all_wstats)
    }
    if normalize != 'total':
        del stat_types['cases']

    correct_cost = efficiency_config.get('correct', 0.0)
    misplaced_cost = efficiency_config.get('misplaced', 0.2)
    duplicates_cost = efficiency_config.get('duplicates', 0.3)
    extra_cost = efficiency_config.get('extra', 0.5)
    missing_cost = efficiency_config.get('missing', 1.0)
    error_cost = correct_cost*all_cstats['correct'] + misplaced_cost*all_cstats['misplaced'] + \
        duplicates_cost*all_cstats['duplicates'] + extra_cost*all_cstats['extra'] + missing_cost*all_cstats['missing']
    efficiency = 1 - error_cost/all_cstats['total_ref'] if all_cstats.get('total_ref', 0) > 0 else 1.0
    #print(f'{stat_types = }')

    for tab_id, (tab_name, stats) in stat_types.items():
        active_class = 'active' if tab_id == 'entries' else ''
        html_parts.extend([
            f'<div id="tab-{tab_id}" class="tab-content {active_class}">',
            f'<h3>Общая статистика по всем ситуациям ({tab_name.lower()})</h3>',
            '<table>',
            '<tr><th>Тип</th><th>Количество</th><th>Процент</th></tr>'
        ])

        for kind in ['correct', 'misplaced', 'duplicates', 'extra', 'missing', normalize]:
            ru_kind = {
                'correct': 'Верные',
                'extra': 'Лишние',
                'misplaced': 'Смещённые',
                'duplicates': 'Дубликаты',
                'missing': 'Пропущенные',
                normalize: 'Всего'
            }.get(kind, kind)

            value = stats.get(kind, 0)
            #print(f'{stats = }')

            percent = value / stats[normalize] if stats.get(normalize, 0) > 0 and kind != normalize else 1.0

            # Форматирование числовых значений
            if tab_id == 'weighted':
                value_str = f'{value:.2f}'
            else:
                value_str = str(value)

            extra = (kind == 'extra' or kind == 'duplicates') and normalize == 'total_ref'
            plus = '+' if extra else ''
            color = ' style=\"color:#C00\"' if extra and normalize == 'total_ref' else ''

            html_parts.append(
                f"<tr><td>{ru_kind}</td>"
                f"<td align='right'>{value_str}</td>"
                f"<td align='right'{color}>{plus}{percent:.1%}</td>"
                f"</tr>"
            )

        html_parts.append('</table></div>')

    # Добавим строку с эффективностью
    html_parts.append(
        f'<p>Эффективность разметки: <strong>{efficiency:.0%}</strong> (стоимость ошибок: {error_cost:.2f}, кодов в эталонной разметке: {all_cstats["total_ref"]} )</p>'
    )

    # Таблица сравнения ситуаций для каждого типа статистики
    for tab_id, (tab_name, _) in stat_types.items():
        active_class = 'active' if tab_id == 'entries' else ''
        html_parts.extend([
            f'<div id="tab-compare-{tab_id}" class="tab-content {active_class}">',
            f'<h3>Сравнительная таблица ситуаций ({tab_name.lower()})</h3>',
            '<table>',
            '<tr><th>Ситуация</th>'
            '<th>Верных</th>'
            '<th>Смещённых</th>'
            '<th>Дубликатов</th>'
            '<th>Лишних</th>'
            '<th>Пропущенных</th>'
            '<th>Эффективность</th>'
            '</tr>'
        ])

        for title, result in situation_results.items():
            stats_dict = {
                'cases': result['stats'],
                'entries': result['cstats'],
                'weighted': result['wstats']
            }
            stats = stats_dict[tab_id]

            # Форматирование числовых значений
            #if tab_id == 'weighted':
            format_val = lambda v: f'{v:.1%}'
            #else:
            #    format_val = lambda v: str(v)
            ccstats = result['cstats']
            case_error_cost = (
                correct_cost * ccstats['correct'] +
                misplaced_cost * ccstats['misplaced'] +
                duplicates_cost * ccstats['duplicates'] +
                extra_cost * ccstats['extra'] +
                missing_cost * ccstats['missing']
            )
            efficiency = 1 - case_error_cost / ccstats['total_ref'] if ccstats['total_ref'] > 0 else 0

            den = stats.get(normalize, 0)
            stats_percent = {k: v / den for k, v in stats.items() if k != normalize} if den > 0 else {}

            extra = normalize == 'total_ref'
            plus = '+' if extra else ''
            color = ' style=\"color:#C00\"' if extra and normalize == 'total_ref' else ''

            html_parts.append(
                f"<tr><td><a href='#situation-{make_safe_id(title)}'>{title}</a></td>"
                f"<td align='right'>{format_val(stats_percent.get('correct', 0))}</td>"
                f"<td align='right'>{format_val(stats_percent.get('misplaced', 0))}</td>"
                f"<td align='right'{color}>{plus}{format_val(stats_percent.get('duplicates', 0))}</td>"
                f"<td align='right'{color}>{plus}{format_val(stats_percent.get('extra', 0))}</td>"
                f"<td align='right'>{format_val(stats_percent.get('missing', 0))}</td>"
                f"<td align='right'>{efficiency:.0%}</td>"
                f"</tr>"
            )

        html_parts.append('</table></div>')

    html_parts.append('</section>')

    # Обновим JavaScript для переключения между таблицами сравнения
    html_parts.append('<script>document.addEventListener("DOMContentLoaded", function() {')
    html_parts.append('function switchTab(tabType) {')
    html_parts.append('  // Обновляем активную кнопку')
    html_parts.append('  document.querySelectorAll(".tab-button").forEach(btn => {')
    html_parts.append('    btn.classList.remove("active");')
    html_parts.append('  });')
    html_parts.append('  document.getElementById("btn-" + tabType).classList.add("active");')
    html_parts.append('  ')
    html_parts.append('  // Обновляем видимый контент таблиц')
    html_parts.append('  document.querySelectorAll(".tab-content").forEach(content => {')
    html_parts.append('    content.classList.remove("active");')
    html_parts.append('  });')
    html_parts.append('  document.getElementById("tab-" + tabType).classList.add("active");')
    html_parts.append('  document.getElementById("tab-compare-" + tabType).classList.add("active");')
    html_parts.append('};')
    html_parts.append('window.switchTab = switchTab;')
    html_parts.append('});</script>')

    # Детальный отчет по каждой ситуации
    html_parts.append('<h2>Детальный анализ ситуаций</h2>')

    for i, (title, result) in enumerate(situation_results.items()):
        html_parts.extend([
            f'<div class="situation-card" id="situation-{make_safe_id(title)}">',
            f'<div class="situation-header" onclick="toggleSituation({i})">',
            f'<h3>{title}' # (Верных: {result["stats"]["correct"]}, Ошибок: {result["stats"]["total"] - result["stats"]["correct"]})'
            f'</h3>',
            '</div>',
            f'<div id="content-{i}" class="situation-content">'
        ])

        # Генерируем отчет для текущей ситуации
        situation_html = generate_html(
            result['text'],
            result['stats'],
            result['cstats'],
            result['wstats'],
            result['errors'],
            efficiency_config=efficiency_config,
            normalize=normalize
        )

        # Извлекаем только содержимое body без заголовков
        body_content = re.search(r'<body>(.*?)</body>', situation_html, re.DOTALL)
        if body_content:
            content = re.sub(r'<h1>.*?</h1>', '', body_content.group(1))
            html_parts.append(content)

        html_parts.append('</div></div>')

    html_parts.append('</div></div></div></body></html>')

    return "\n".join(html_parts)


def generate_multi_situation_report_text(situations_dict, reference_dict, ignore_text_errors=False, ignore_codes=None):
    """
    Создает HTML-отчет для нескольких ситуаций с общей сводкой и детализацией

    :param situations_dict: словарь {название: разметка_ситуации}
    :param reference_dict: словарь {название: эталонная_разметка}
    :return: строка -- сводная таблица в формате Markdown
    """
    # Собираем статистику по всем ситуациям
    situation_results = {}
    all_stats = {}
    all_cstats = {}
    all_wstats = {}

    not_in_reference = []
    for title, test_text in situations_dict.items():
        if title not in reference_dict:
            print(f"Предупреждение: '{title}' не найден в эталонных данных.")
            not_in_reference.append(title)
            continue
        reference_text = reference_dict.get(title)
        stats, cstats, wstats, errors, text_mismatch_errors = compare_markups(
            reference_text, test_text, ignore_text_errors=True, situation_title=title, ignore_codes=ignore_codes
        )
        situation_results[title] = {
            'stats': stats,
            'cstats': cstats,
            'wstats': wstats,
            'errors': errors,
            'text': test_text
        }

        # Объединяем статистику
        all_stats = add_dicts(all_stats, stats) if all_stats else stats.copy()
        all_cstats = add_dicts(all_cstats, cstats) if all_cstats else cstats.copy()
        all_wstats = add_dicts(all_wstats, wstats) if all_wstats else wstats.copy()

    if not situation_results:
        print("Нет ситуаций для анализа.")
        return ""

    # Формируем текстовый отчет (общий)
    header = ["", "Случаи", "Вхождения", "Взвешенные вхождения"]
    ru_map = {
        'correct': 'Верные',
        'misplaced': 'Смещённые',
        'duplicates': 'Дубликаты',
        'extra': 'Лишние',
        'missing': 'Пропущенные'
    }
    table = [header]
    for kind in ['correct', 'misplaced', 'duplicates', 'missing', 'extra']:
        row = [ru_map[kind]]
        for stats in [all_stats, all_cstats, all_wstats]:
            value = stats.get(kind, 0)/stats['total'] if stats['total'] > 0 else 0
            row.append(f'{value:.1%}')
        table.append(row)
    widths = [max(len(str(item)) for item in col) for col in zip(*table)]
    table_str = ""
    # Определяем символы для рисования Unicode-таблицы
    h_line = "─"  # горизонтальная линия
    v_line = "│"  # вертикальная линия
    tl_corner = "┌"  # верхний левый угол
    tr_corner = "┐"  # верхний правый угол
    bl_corner = "└"  # нижний левый угол
    br_corner = "┘"  # нижний правый угол
    t_down = "┬"  # T-образное соединение сверху
    t_up = "┴"  # T-образное соединение снизу
    t_right = "├"  # T-образное соединение слева
    t_left = "┤"  # T-образное соединение справа
    cross = "┼"  # перекрестие

    # Рисуем таблицу
    for i, row in enumerate(table):
        if i == 0:
            # Верхняя граница
            top_border = tl_corner
            for j, width in enumerate(widths):
                top_border += h_line * (width + 2)
                top_border += tr_corner if j == len(widths) - 1 else t_down
            table_str += top_border + "\n"

        # Строка с данными
        data_line = v_line
        for j, item in enumerate(row):
            align = "<" if j == 0 else ">"
            data_line += f" {str(item):{align}{widths[j]}} "
            data_line += v_line
        table_str += data_line + "\n"

        # Горизонтальные разделители
        if i == 0 or i == len(table) - 1:
            separator = t_right if i == 0 else bl_corner
            for j, width in enumerate(widths):
                separator += h_line * (width + 2)
                if j == len(widths) - 1:
                    separator += t_left if i == 0 else br_corner
                else:
                    separator += cross if i == 0 else t_up
            table_str += separator + "\n"

    return table_str


def read_situations_from_file(filename, fix_codes=False):
    """
    Читает файл с множеством ситуаций и разбивает их на словарь.
    Каждая ситуация должна начинаться с заголовка формата "# <название> \n"

    Возвращает dict: название ситуации -> текст ситуации (включая заголовок)
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        if fix_codes:
            # Добавляем пропущенные `*` перед кодами в тестируемой разметке
            content = fix_codes_syntax(content)

    # Разбиваем файл на отдельные ситуации по заголовкам
    situations = re.split(r'(?=^# .*$)', content, flags=re.MULTILINE)

    def remove_redundant(text):
        # Удаляем первую строку с названием
        text = re.sub(r'^# .*?\n', '', text, count=1, flags=re.MULTILINE)
        # Удаляем лишние пробелы и переносы строк
        text = text.strip()
        return text

    # Удаляем пустые строки, если они есть
    situations = [s.strip() for s in situations if s.strip()]

    # Создаем словарь: название -> текст
    result = {}
    for situation in situations:
        # Ищем название в первой строке
        match = re.match(r'^# (.*?)$', situation, flags=re.MULTILINE)
        if match:
            title = match.group(1).strip()
            # Удаляем двойные пробелы
            title = re.sub(r'\s+', ' ', title)
            if title in result:
                # Обрабатываем дубликаты названий
                i = 1
                while f"{title}_{i}" in result:
                    i += 1
                title = f"{title}_{i}"
            result[title] = remove_redundant(situation)

    return result


def gen_compare_report(ref='reference.txt', tested='test_multi.txt', html=True):
    ref_text = read_situations_from_file(ref)
    test_text = read_situations_from_file(tested)
    if html:
        generate = generate_multi_situation_report(test_text, ref_text)
        with open('report_multi.html', 'w', encoding='utf-8') as f:
            f.write(generate)
        print("Готово: см. report_multi.html")
    else:
        generate = generate_multi_situation_report_text(test_text, ref_text)
        print(generate)


def main():
    """
    Основная функция для запуска сравнения разметки из командной строки
    с профессиональным интерфейсом
    """
    parser = argparse.ArgumentParser(
        description='Сравнение разметки ситуаций и генерация отчета',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # тестируемый файл (обязательный, позиционный аргумент)
    parser.add_argument(
        'filename',
        help='Путь к файлу с тестируемой разметкой',
        type=str
    )

    parser.add_argument(
        '-r', '--reference',
        help='Путь к файлу с эталонной разметкой',
        default='reference.txt',
        type=str
    )

    parser.add_argument(
        '-o', '--output',
        help='Путь для сохранения HTML-отчета',
        default='report.html',
        type=str
    )

    parser.add_argument(
        '--text-only',
        help='Вывести только текстовый отчет без генерации HTML',
        action='store_true'
    )

    parser.add_argument(
        '--both',
        help='Вывести текстовый отчет и сгенерировать HTML',
        action='store_true'
    )

    parser.add_argument(
        '--efficiency-config',
        help='Путь к файлу с конфигурацией эффективности (JSON)',
        default='efficiency_config.json',
        type=str
    )

    parser.add_argument(
        '--normalize',
        help='Способ нормализации статистики: "total" (по всем случаям), "ref" (по случаям из эталона), "test" (по случаям из тестируемой разметки)',
        choices=['total', 'ref', 'test'],
        default='ref',
        type=str
    )

    parser.add_argument(
        '--ignore-errors',
        help='Не выводить ошибки сопоставления текстов между эталоном и тестируемой разметкой',
        action='store_true'
    )

    parser.add_argument(
        '--fix-codes',
        help='Добавлять пропущенные `*` перед кодами в тестируемой разметке',
        action='store_true'
    )

    # Новый режим экспорта в JSON
    parser.add_argument(
        '--json-only',
        help='Сгенерировать только JSON-отчет и не генерировать HTML/текст',
        action='store_true'
    )
    parser.add_argument(
        '--json-output',
        help='Путь для сохранения JSON-отчета',
        default='report.json',
        type=str
    )
    
    # Список кодов для игнорирования
    parser.add_argument(
        '--ignore',
        help='Список кодов для игнорирования (через запятую), например: C1,C2',
        default='',
        type=str
    )

    args = parser.parse_args()

    try:
        efficiency_config = json.load(open(args.efficiency_config, 'r', encoding='utf-8'))
    except json.JSONDecodeError as e:
        print(f"Ошибка чтения файла конфигурации эффективности: {e}")
        return 1
    except FileNotFoundError:
        print(f"Ошибка: Файл с конфигурацией эффективности '{args.efficiency_config}' не найден. Используется конфигурация по умолчанию.")
        efficiency_config = None

    # Проверка существования входных файлов
    if not os.path.exists(args.reference):
        print(f"Ошибка: Файл с эталонной разметкой '{args.reference}' не найден")
        return 1

    if not os.path.exists(args.filename):
        print(f"Ошибка: Файл с тестируемой разметкой '{args.filename}' не найден")
        return 1

    # Загрузка данных
    print(f"Загрузка эталонной разметки из файла: {args.reference}")
    ref_text = read_situations_from_file(args.reference, fix_codes=args.fix_codes)
    print(f"Загружено {len(ref_text)} эталонных ситуаций")

    print(f"Загрузка тестируемой разметки из файла: {args.filename}")
    test_text = read_situations_from_file(args.filename, fix_codes=args.fix_codes)
    print(f"Загружено {len(test_text)} тестируемых ситуаций")

    normalize = {'total': 'total',
                 'ref': 'total_ref',
                 'test': 'total_test'}[args.normalize]

    # Разбор списка игнорируемых кодов
    ignore_codes = set()
    if args.ignore:
        tokens = re.split(r'[\,\s]+', args.ignore.strip())
        for token in tokens:
            if not token:
                continue
            code = map_eng(token.strip())
            if not code.startswith('*'):
                code = '*' + code
            ignore_codes.add(code)

    # Если выбран только JSON-режим — генерируем JSON и выходим
    if args.json_only:
        print("\nГенерация JSON-отчета...")
        json_report = generate_multi_situation_report_json(
            test_text,
            ref_text,
            efficiency_config=efficiency_config,
            normalize=normalize,
            ignore_text_errors=args.ignore_errors,
            ignore_codes=ignore_codes
        )
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)
        print(f"JSON-отчет сохранен в: {args.json_output}")
        return 0

    # Генерация отчетов
    if args.text_only or args.both:
        print("\nГенерация текстового отчета:")
        text_report = generate_multi_situation_report_text(test_text, ref_text, args.ignore_errors, ignore_codes)
        print(text_report)

    if not args.text_only or args.both:
        print("\nГенерация HTML-отчета...")
        html_report = generate_multi_situation_report(test_text, ref_text, efficiency_config=efficiency_config,
                                                      normalize=normalize, ignore_text_errors=args.ignore_errors, ignore_codes=ignore_codes)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"HTML-отчет сохранен в: {args.output}")

    return 0


def generate_multi_situation_report_json(situations_dict, reference_dict, normalize='total', efficiency_config=None, ignore_text_errors=False, ignore_codes=None):
    """
    Строит JSON-отчет по множеству ситуаций.

    :param situations_dict: словарь {название: разметка_ситуации}
    :param reference_dict: словарь {название: эталонная_разметка}
    :param normalize: ключ нормализации ('total' | 'total_ref' | 'total_test')
    :param efficiency_config: словарь весов ошибок {'correct','misplaced','duplicates','extra','missing'}
    :param ignore_text_errors: если False — включать сообщения о несоответствии текстов
    :return: словарь, пригодный для сериализации в JSON
    """
    efficiency_config = efficiency_config or {}

    correct_cost = efficiency_config.get('correct', 0.0)
    misplaced_cost = efficiency_config.get('misplaced', 0.2)
    duplicates_cost = efficiency_config.get('duplicates', 0.3)
    extra_cost = efficiency_config.get('extra', 0.5)
    missing_cost = efficiency_config.get('missing', 1.0)

    def serialize_errors_per_block(errors_list):
        # errors_list: список из 7 словарей (по блокам 0..6)
        serialized_blocks = []
        for block_errors in errors_list:
            block_serialized = {}
            for kind, items in block_errors.items():
                out_items = []
                for it in items:
                    code = it.get('code')
                    expected = it.get('expected', [])
                    found = it.get('found', [])
                    if isinstance(expected, set):
                        expected = sorted(expected)
                    if isinstance(found, set):
                        found = sorted(found)
                    out_items.append({'code': code, 'expected': expected, 'found': found})
                block_serialized[kind] = out_items
            serialized_blocks.append(block_serialized)
        return serialized_blocks

    situation_results = {}
    all_stats = {}
    all_cstats = {}
    all_wstats = {}
    all_text_errors = []
    not_in_reference = []

    for title, test_text in situations_dict.items():
        if title not in reference_dict:
            not_in_reference.append(title)
            continue

        reference_text = reference_dict.get(title)
        try:
            stats, cstats, wstats, errors, text_mismatch_errors = compare_markups(
                reference_text, test_text, ignore_text_errors, title, ignore_codes
            )
        except Exception as e:
            situation_results[title] = {
                'error': str(e)
            }
            continue

        if text_mismatch_errors and not ignore_text_errors:
            all_text_errors.extend(text_mismatch_errors)

        errors_json = serialize_errors_per_block(errors)

        case_error_cost = (
            correct_cost * cstats.get('correct', 0)
            + misplaced_cost * cstats.get('misplaced', 0)
            + duplicates_cost * cstats.get('duplicates', 0)
            + extra_cost * cstats.get('extra', 0)
            + missing_cost * cstats.get('missing', 0)
        )
        case_total_ref = cstats.get('total_ref', 0) or 0
        case_efficiency = 1 - case_error_cost / case_total_ref if case_total_ref > 0 else 0.0

        situation_results[title] = {
            'stats': stats,
            'cstats': cstats,
            'wstats': wstats,
            'errors': errors_json,
            'efficiency': case_efficiency
        }

        all_stats = add_dicts(all_stats, stats) if all_stats else stats.copy()
        all_cstats = add_dicts(all_cstats, cstats) if all_cstats else cstats.copy()
        all_wstats = add_dicts(all_wstats, wstats) if all_wstats else wstats.copy()

    overall_error_cost = (
        correct_cost * all_cstats.get('correct', 0)
        + misplaced_cost * all_cstats.get('misplaced', 0)
        + duplicates_cost * all_cstats.get('duplicates', 0)
        + extra_cost * all_cstats.get('extra', 0)
        + missing_cost * all_cstats.get('missing', 0)
    ) if all_cstats else 0.0
    overall_total_ref = all_cstats.get('total_ref', 0) if all_cstats else 0
    overall_efficiency = 1 - overall_error_cost / overall_total_ref if overall_total_ref > 0 else 0.0

    result = {
        'normalize': normalize,
        'costs': {
            'correct': correct_cost,
            'misplaced': misplaced_cost,
            'duplicates': duplicates_cost,
            'extra': extra_cost,
            'missing': missing_cost
        },
        'overall': {
            'stats': all_stats,
            'cstats': all_cstats,
            'wstats': all_wstats,
            'efficiency': overall_efficiency
        },
        'situations': situation_results,
        'not_in_reference': not_in_reference
    }

    if not ignore_text_errors:
        result['text_mismatch_errors'] = all_text_errors
        result['text_mismatch_errors_count'] = len(all_text_errors)

    return result


if __name__ == "__main__":
    exit(main())

