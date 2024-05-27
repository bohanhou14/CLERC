import os
import re
import sys
import numpy as np
import pandas as pd
from eyecite import get_citations, clean_text, resolve_citations
from eyecite.models import FullCaseCitation, Resource, CaseCitation
from tqdm import tqdm, trange

dd = os.environ['DATADIR']
def clean_cstr(c):
    # remove spaces and dots
    l = re.sub(r'\W+', '', c)
    return l

def eyecite_extract_citations(text):
    if text == "":
        return [], []
    # print(text)
    text = clean_text(text, ['html', 'all_whitespace'])
    # tokenizer = HyperscanTokenizer(cache_dir='.test_cache')
    citations = get_citations(text)
    citations = list(citations)
    resolutions = resolve_citations(citations)
    cases = [res for res in resolutions if (isinstance(res.citation, FullCaseCitation) or isinstance(res.citation, CaseCitation))]
    case_labels = [c.citation.token.data for c in cases]
    case_pos = [(c.citation.token.start, c.citation.token.end) for c in cases]
    return case_labels, case_pos

def find_all_exact_p(text, p):
    res = []
    for m in re.finditer(re.escape(p), text):
        res.append((m.start(), m.end()))
    return res

def find_all_p(text, p):
    res = []
    for m in re.finditer(p, text):
        res.append((m.start(), m.end()))
    return res
# text = "Matsushita Elec. Indus. Co. v. Zenith Radio Corp.,  475U.S.574 , 586-87, 106 S.Ct. 1348, 89 L.Ed.2d 538 (1986)."
# citation = "106 S.Ct. 1348"

# sliding window compare the first 2 chars to determine if it is an abbreviation
def is_abbr(abbr_words, full_words):
    is_abbrev = True
    for i in range(min(len(abbr_words), len(full_words))):
        if len(abbr_words[i]) == 0 or len(full_words[i]) == 0:
            is_abbrev = False
            break
        elif abbr_words[i][0] != full_words[i][0]:
            is_abbrev = False
            break
    if is_abbrev == False:
        # breakpoint()
        dot_split = [w.split(".") for w in abbr_words]
        for i in range(len(dot_split)):
            d = dot_split[i]
            for w in d:
                if w == "":
                    d.remove(w)
            dot_split[i] = d

        for i in range(len(dot_split)):
            curr_w_list = dot_split[i]
            if len(curr_w_list) == len(full_words):
                try:
                    all_true = [True for j in range(len(curr_w_list)) if len(curr_w_list[j]) > 0 and (curr_w_list[j][0] == full_words[j][0])]
                except:
                    print(f"curr_w_list: {curr_w_list}")
                    print(f"full_words: {full_words}")
                    assert False
                if all(all_true):
                    is_abbrev = True
                    break
        return is_abbrev
    return is_abbrev

def list_rindex(ls, val):
    try:
        ridx = len(ls) - ls[-1::-1].index(val) -1
    except:
        return -1
    return ridx

def find_citation(citation, text, meta_cleaned_cites, name_abbrevs, years):
    # print(citation)
    citation_start = text.find(citation)
    if citation_start == -1:
        print(f"Cannot find citation: {citation}")
        return text, None
    cleaned_cstr = clean_cstr(citation)
    found_index = -1
    for i in range(len(meta_cleaned_cites)):
        if cleaned_cstr in meta_cleaned_cites[i]:
            # print(f"Found {cleaned_cstr} in index = {i}")
            found_index = i
            # print(meta_cites[i])
            # print(meta_cleaned_cites[i])
            break
    start_index = -1   
    # breakpoint()     
    if found_index != -1:
        name = name_abbrevs[found_index]
        start_index = text.find(name)
        # logic:
        # find citation name (entity name) first. If name not found, 
        # it is possible that the name is abbreviated or different by human typos.
        # Then find "v." and split the name into two entities, and find the index of the first entity

        if start_index == -1:
            # breakpoint()
            name_words = name.split(" ")
            try:
                vindex = name_words.index("v.")
            except:
                vindex = -1    
            if vindex != -1:
                entity_A = name_words[0:vindex]
                entity_B = name_words[vindex+1:]
                # find the index of the first word of entity A
                words = text.split(" ")
                before_text = text[:citation_start]
                before_words = before_text.split(" ")
                before_words_len = len(before_words)
                try:
                    # find the nearest index of v. before citation_start
                    # but it is possible that this v. does not correspond to the citation we are looking for
                    vindex_in_words = list_rindex(before_words, "v.")
                except:
                    vindex_in_words = -1

            if vindex != -1 and vindex_in_words != -1:
                entity_A_candidate = before_words[vindex_in_words-len(entity_A):vindex_in_words]
                # sanity check we are extracting the correct entity A
                # by comparing if the first letters of each word are the same
                # breakpoint()
                if is_abbr(entity_A_candidate, entity_A):
                    entity_A_in_words = entity_A_candidate
                    entity_A_in_text = " ".join(entity_A_in_words)
                    # breakpoint()
                    # abbreviation of entity A will still have the same number of white spaces as the full name
                    start_index = text.index(entity_A_in_text)
                else:
                    print("Cannot extract entity A by v.")
                    print("Continuing!")

            # if vindex_in_words is -1, then v. is not found in the text
            # it can be a short form citation, or In re, Id., see, etc.
            else:
                b_id_start = text[:citation_start].rfind("Id.")
                s_id_start = text[:citation_start].rfind("id.")
                b_see_start = text[:citation_start].rfind("See")
                s_see_start = text[:citation_start].rfind("see")
                b_inre_start = text[:citation_start].rfind("In re ")
                s_inre_start = text[:citation_start].rfind("in re ")
                id_start = max(b_id_start, s_id_start)
                see_start = max(b_see_start, s_see_start)
                inre_start = max(b_inre_start, s_inre_start)
                if id_start == -1 and see_start == -1 and inre_start == -1:
                    print("Cannot find citation, probably wrong text")
                    print((text, citation))
                    start_index = -1
                    return text, None
                # below two cases discuss if one of the two cases exists
                else:
                    start_index = max(max(see_start, id_start), inre_start)
                    start_words = text[start_index:citation_start].split(" ")
                    # if there are too many words between the start and the citation, probably wrong!
                    if len(start_words) > 10:
                        start_index = -1
                        print("Cannot find citation, probably wrong text")
                        print(text, citation)
                        return text, None

    # Next, we find the end index of the citation sentence
    # Most citations have the year, so we find the year first
    # breakpoint()
    year_index = text[start_index:].find(years[found_index]) + start_index if text[start_index:].find(years[found_index]) != -1 else -1
    # breakpoint()
    # find the next period after the year
    if year_index != -1:
        period_index = text[year_index:].find(".") + year_index if text[year_index:].find(".") != -1 else sys.maxsize
        semicolon_index = text[year_index:].find(";") + year_index if text[year_index:].find(";") != -1 else sys.maxsize
        comma_index = text[year_index:].find("),") + year_index if text[year_index:].find("),") != -1 else sys.maxsize
    else:
        # breakpoint()
        # if this citation does not have a year, then we find the next period after all the citations in the sequence
        citation_end = citation_start+len(citation)
        last_found_end = citation_end
        # find the other citations in this citation sentence:
        other_cites, other_pos = eyecite_extract_citations(text[citation_end:])
        cleaned_other_cites = [clean_cstr(c) for c in other_cites]
        # breakpoint()
        for c in meta_cleaned_cites[found_index]:
            if c in cleaned_other_cites:
                other_cite_index = cleaned_other_cites.index(c)
                other_cite_pos = other_pos[other_cite_index]
                curr_found_end = other_cite_pos[1] + citation_end
                if curr_found_end > last_found_end:
                    last_found_end = curr_found_end
            # curr_found_end = text[citation_end:].find(c) + citation_end + len(c) if text[citation_end:].find(c) != -1 else -1
            # if curr_found_end > last_found_end:
            #     last_found_end = curr_found_end
        period_index = text[last_found_end:].find(".") + citation_end if text[last_found_end:].find(".") != -1 else sys.maxsize
        semicolon_index = text[last_found_end:].find(";") + citation_end if text[last_found_end:].find(";") != -1 else sys.maxsize
        comma_index = text[last_found_end:].find("),") + citation_end if text[last_found_end:].find("),") != -1 else sys.maxsize
        comma_index = text[last_found_end:].find(").") + citation_end if text[last_found_end:].find(").") != -1 else sys.maxsize
    # the closest period or semicolon is the end of the citation sentence
    end_index = min(min(period_index, semicolon_index), comma_index)
    # plus the white space after the period or semicolon
    citation_sent = text[start_index:end_index+2]
    text = text.replace(citation_sent, " REDACTED ")
    return text, citation_sent

fdq = chr(8220)
bdq = chr(8221)
quotes = [fdq, bdq, '"',]
quote_w = ['quoting', 'quote', 'quotes', 'quoted']
preds = []

def classify_chained_quote(text, big_window=15, small_window = 10):
    words = text.split(" ")
    mid = int(len(words) / 2) # because the middle of the query is the central citation

    # only search the left half of the text
    bleft = words[mid-big_window:mid]
    bright = words[mid:mid + big_window]
    sleft = words[mid - small_window: mid]
    sright = words[mid:mid + small_window]
    bwindow = words[mid-big_window:mid + big_window]
    
    def quote_w_in_w(w):
        for q in quote_w:
            if q in w.lower():
                return True
        return False
    def quote_in_w(w):
        for q in quotes:
            if q in w:
                return True
        return False

    for w in sleft:
        if quote_w_in_w(w):
            return 1
    
    for w in sright:
        if quote_in_w(w) or quote_w_in_w(w):
            return 1

    semicolon_idx = -1
    quote_idx = -1
    for i in range(len(bleft)):
        w = bleft[i]
        # # if the quote gets long, then still possible in the big left window
        # if quote_w_in_w(w):
        #     return 1
        if ";" in w:
            semicolon_idx = i
        if quote_in_w(w):
            quote_idx = i  
        # if the quote mark occurs before the semicolon, then it's not a direct quote
        if (semicolon_idx > 0) and (quote_idx > 0) and (quote_idx < semicolon_idx):
            return 0
        # otherwise, if the quote mark occurs after the semicolon, 
        # or the quote exists without the semicolon,
        # then it's a direct quote
        if quote_idx > 0:
            if mid - quote_idx < small_window:
                return 1

    semicolon_idx = -1
    for i in range(len(bright)):
        w = bright[i]
        if ';' in w:
            semicolon_idx = i
        # quote must be before the semicolon
        if (semicolon_idx == -1) and (quote_in_w(w) or quote_w_in_w(w)):
            return 1
    return 0, ""

# def classify_direct_quote(text, big_window=200, small_window=100):
#     mid = len(text) // 2 # because the middle of the query is the central citation
#     # only search the left half of the text
#     def find_all_p(text, p):
#         res = []
#         for m in re.finditer(p, text):
#             res.append((m.start(), m.end()))
#         return res
#     bleft = mid-big_window
#     bright = mid+big_window
#     sleft = mid-small_window
#     sright = mid+small_window
#     def in_interval(num, start, end):
#         return (num >= start) and (num <= end)
#     p = f'(?<={fdq})(.*?)(?={bdq})'
#     quoted_pos = find_all_p(text, p)
#     most_likely_candidate = ""
#     confidence = -1
#     for pos in quoted_pos:
#         # least likely to be a direct quote
#         cand = text[pos[0]-1:pos[1]+1]
#         if len(cand.split(" ")) <= 5:
#             continue
#         if in_interval(pos[0], sleft, mid) and in_interval(pos[1], mid, sright):
#             most_likely_candidate = cand
#             confidence = 4
#             break
#         elif in_interval(pos[0], mid, bright):
#             if confidence < 1:
#                 most_likely_candidate = cand
#                 confidence = 1
#         elif (in_interval(pos[0], bleft, mid) and in_interval(pos[1], bleft, mid)) or (in_interval(pos[0], mid, bright) and in_interval(pos[1], mid, bright)):
#             if confidence < 3:
#                 most_likely_candidate = cand
#                 confidence = 3
#         elif in_interval(pos[0], bleft, mid) and in_interval(pos[1], mid, bright):
#             if confidence < 4:
#                 most_likely_candidate = cand
#                 confidence = 4
#     if most_likely_candidate != "":
#         return 1, most_likely_candidate
#     else:
#         return 0, most_likely_candidate

def classify_direct_quote(text, cite, window=150):

    # in case there are multiple cites, we take the only closest to the center of the text
    mid = len(text) // 2
    # regex find multiple cites
    cites = find_all_exact_p(text, cite)

    # find the distance from the center of the text to the cite
    dis = [abs(c[0]-mid) for c in cites]
    # get the closest cite
    closest_cite_start = cites[dis.index(min(dis))][0]
    closest_cite_end = cites[dis.index(min(dis))][1]

    # find the closest sentence in double quote to the cite
    p = f'(?<={fdq})(.*?)(?={bdq})'
    quotes_dis = find_all_p(text, p)
    if len(quotes_dis) == 0:
        return None

    # if len(quotes_dis) == 0:
    #     p = f'(?<=")(.*?)(?=")'
    #     quotes_dis = find_all_p(text, p)
    
    # find the distance from the center of the text to the closest quote
    # if the sentence occurs before the cite
    start_dis = [abs(c[1] - closest_cite_start) for c in quotes_dis]
    # if the sentence occurs after the cite
    end_dis = [abs(c[0] - closest_cite_end) for c in quotes_dis]

    if min(start_dis) > window and min(end_dis) > window:
        return None

    # get the left quote
    left_quote_s = quotes_dis[start_dis.index(min(start_dis))][0]
    left_quote_e = quotes_dis[start_dis.index(min(start_dis))][1]
    left_quote = text[left_quote_s:left_quote_e]
    # get the right quote
    right_quote_s = quotes_dis[end_dis.index(min(end_dis))][0]
    right_quote_e = quotes_dis[end_dis.index(min(end_dis))][1]
    right_quote = text[right_quote_s:right_quote_e]

    quote = left_quote if min(start_dis) < min(end_dis) else right_quote
    # if the quote is too short, then ignore
    if len(quote.split(" ")) < 5:
        return None
    # return whichever is the closest quote
    return quote

def clean_incomplete_removals(extracted_text):
    def remove_patterns(text, regex):
        patterns = re.findall(regex, text)
        if len(patterns) > 0:
            for p in patterns:
                text = text.replace(p, "")
        return text
    # remove years that incomplete removals
    extracted_text = remove_patterns(extracted_text, r'\([^(]*\d{4}\)')

    # remove citations that have incomplete removals
    # Federal Reporters
    extracted_text = remove_patterns(extracted_text, r'\d+ U.S. \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ U. S. \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ S.Ct. \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ L.Ed. \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ L.Ed.\d+d \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ L. Ed. \d+d \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ F. \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ F.2d \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ F.3d \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ F.Supp. \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ F.Supp. 2d \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ F.Supp. 3d \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ F. Supp. \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ F. Supp. 2d \d*')
    extracted_text = remove_patterns(extracted_text, r'\d+ F. Supp. 3d \d*')
    return extracted_text

def filter_queries(queries_path, num_queries=1000, option="single", debug=False):
    # initializing the metadata
    df = pd.read_csv(f"{dd}/case.law.data/metadata.csv", sep=",")
    meta_cites = df['cites'].tolist()
    meta_cites = [c.replace('"', "") for c in meta_cites]
    meta_cites_list = [c.split(";") for c in meta_cites]
    for i in range(len(meta_cites_list)):
        for j in range(len(meta_cites_list[i])):
            meta_cites_list[i][j] = meta_cites_list[i][j].strip()
    # breakpoint()
    meta_cleaned_cites = []
    for cites in tqdm(meta_cites_list, desc="Cleaning citations"):
        meta_cleaned_cites.append([clean_cstr(c) for c in cites])

    name_abbrevs = df['name_abbreviation'].tolist()
    decision_date = df['decision_date_original'].tolist()
    # parse the year from 1996-01-24 using regex
    years = [re.findall(r'\d{4}', d)[0] for d in decision_date]

    # process the queries
    df = pd.read_csv(queries_path, sep="\t", names=['qid', 'text', 'citation', 'label', 'direct_quote'])
    df = df.replace(np.nan, None)
    citations = np.array(df['citation'].tolist())
    texts = np.array(df['text'].tolist())
    labels = np.array(df['label'].tolist())
    qids = np.array(df['qid'].tolist())
    print(texts[0]) # DEBUG
    # quotes = df['quote'].tolist()
    if num_queries == -1:
        num_queries = len(texts)
    direct_queries = [] ; indirect_queries = []
    direct_unextracted_queries = [] ; indirect_unextracted_queries = []
    direct_queries_idx = [] ; indirect_queries_idx = []
    direct_citation_extracted = [] ; indirect_citation_extracted = []
    direct_quote = []
    failed_idx = []

    failed = 0
    for i in trange(num_queries):
        text = texts[i]
        is_failed = False
        extracted_text, citation_sent = find_citation(labels[i], text, meta_cleaned_cites, name_abbrevs, years)
        if citation_sent == None or len(citation_sent) == 0:
            is_failed = True
        # if is all remove version, then we need to remove all citations
        # we still need to first try to remove the central citation and see if it is a failed query
        # if it fails, we will fast forward to the next query
        # this one they are absolutely consistent with the single-removed version
        if option == 'all' and is_failed == False:
            case_labels, case_pos = eyecite_extract_citations(text)
            citation_sent = []
            extracted_text = text
            # breakpoint()
            for j in range(len(case_labels)):
                c = case_labels[j]
                extracted_text, citation_s = find_citation(c, extracted_text, meta_cleaned_cites, name_abbrevs, years)
                if citation_s != None and len(citation_s) >= 0:
                    citation_sent.append(citation_s)
                    # if the sentence cannot be extracted but the label does exist
                    # then let's remove the labels
                    if extracted_text.find(c) != -1:
                        # breakpoint()
                        if debug:
                            print(f"Case Label: {c}")
                            print(f"Case Labels: {case_labels}")
                            print(f"Extracted text: {extracted_text}\n")
                        # remove all the labels (can be more than one) and nearby punctuations as well
                        c_start = extracted_text.find(c)
                        while c_start != -1:
                            c_start = extracted_text.find(c)
                            if c_start == -1:
                                break
                            c_end = min(c_start + len(c), len(extracted_text)-1)
                            if extracted_text[c_end] in [",", ';', '"', "'", ")", "}", "]", ">", ":", "?"]:
                                c_end += 2 # also remove the space following the punctuation
                            if extracted_text[c_start-1] in ['"', ' ', '(', '{', '[', '<', ':', '?']:
                                c_start -= 1
                            # breakpoint()
                            extracted_text = extracted_text[:c_start] + " REDACTED " + extracted_text[c_end:]
                        if debug:
                            print(f"Extracted text after removal: {extracted_text}\n") 
            # remove incomplete removals (such as year and reporter)
            extracted_text = clean_incomplete_removals(extracted_text)    
        
        # dq = quotes[i]
        # dq = eval(dq) if dq != None else dq
        candidate = classify_direct_quote(text, labels[i])

        if is_failed == True:
            failed += 1
            failed_idx.append(i)
        elif candidate != None:
            direct_queries_idx.append(i)
            direct_unextracted_queries.append(text)
            extracted_text = extracted_text.replace(candidate, " REDACTED ")
            direct_queries.append(extracted_text)
            direct_citation_extracted.append(citation_sent)
            direct_quote.append(candidate)
        else:
            indirect_queries_idx.append(i)
            indirect_unextracted_queries.append(text)
            indirect_queries.append(extracted_text)
            indirect_citation_extracted.append(citation_sent)
            
    print("Summary: ")
    print(f"Failed to extract citation in {failed} queries out of {len(direct_queries) + len(indirect_queries) + failed}, which is {failed/(len(direct_queries) + len(indirect_queries) + failed)*100}%")
    print(f"Number of direct queries: {len(direct_queries)}")
    print(f"Number of indirect queries: {len(indirect_queries)}")
    print(f"Percentage of direct queries: {len(direct_queries)/(len(direct_queries) + len(indirect_queries))*100}%")

    direct_data = {
        "qid": direct_queries_idx,
        "text": direct_queries,
        "citation": citations[direct_queries_idx],
        "label": labels[direct_queries_idx],
        "removed_sent": direct_citation_extracted,
        "quote": direct_quote
    }
    direct_unextracted_data = {
        "qid": direct_queries_idx,
        "text": direct_unextracted_queries,
        "citation": citations[direct_queries_idx],
        "label": labels[direct_queries_idx],
        "removed_sent": direct_citation_extracted,
        "quote": direct_quote
    }
    indirect_data = {
        "qid": indirect_queries_idx,
        "text": indirect_queries,
        "citation": citations[indirect_queries_idx],
        "label": labels[indirect_queries_idx],
        "removed_sent": indirect_citation_extracted
    }
    indirect_unextracted_data = {
        "qid": indirect_queries_idx,
        "text": indirect_unextracted_queries,
        "citation": citations[indirect_queries_idx],
        "label": labels[indirect_queries_idx],
        "removed_sent": indirect_citation_extracted
    }
    
    print("Saving direct queries...")
    stripped_queries_path = queries_path.replace(".tsv", "")
    stripped_queries_path += f'-{option}'
    pd.DataFrame(data=direct_data).to_csv(stripped_queries_path+f"-direct_removed-n={num_queries}.tsv" , sep="\t", header=False, index=False)
    pd.DataFrame(data=direct_unextracted_data).to_csv(stripped_queries_path+f"-direct_unremoved-n={num_queries}.tsv", sep="\t", header=False, index=False)
    print("Direct queries saved!")
    print("Saving indirect queries...")
    pd.DataFrame(data=indirect_data).to_csv(stripped_queries_path+f"-indirect_removed-n={num_queries}.tsv", sep="\t", header=False, index=False)
    pd.DataFrame(data=indirect_unextracted_data).to_csv(stripped_queries_path+f"-indirect_unremoved-n={num_queries}.tsv", sep="\t", header=False, index=False)
    print("Indirect queries saved!")

    print("Saving failed queries...")
    failed_data = {
        "qid": failed_idx,
        "text": texts[failed_idx],
        "citation": citations[failed_idx],
        "label": labels[failed_idx]
    }
    pd.DataFrame(data=failed_data).to_csv(stripped_queries_path+f"-failed-n={num_queries}.tsv", sep="\t", header=False, index=False)
    print("Failed queries saved!")
