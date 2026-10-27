# D8 mapping audit — gold title → corpus page (read-only, 2026-10-27)

Scope: the DEV200 gold-title → Wikipedia-page mapping that D8 arm Xp relies on
(freeze docs/EFBPT_D8_REACHABLE_CEILING_FREEZE.md §2–§3). Nothing was run on a
GPU, nothing under outputs/ was written, no repo file was edited. The only
artefact is this file.

Method. A scratch script outside the repo (`python -I`) imported
`build_title_map` from eval/error_analysis_tests/efbpt/d8_reachable_ceiling.py
and applied it to the 613 distinct DEV200 gold titles (427 present, asserted),
then made one more streaming pass over rag/index/wikipedia_full_meta.jsonl to
fetch the first 3 chunks and the total chunk count of every page needed, and to
search norm() containment for the traced-but-absent titles. Inputs:
data/strategyqa_official/efbpt/stage0/lexical_trace_instances.jsonl (636),
data/strategyqa_official/dev200_seed4242.jsonl (200), the corpus (stream only).
All samples: `random.Random(20261027).sample(list_in_file_order, 40)`.
Judgements in §2 and §3 are mine (the model's), made from question_en, the
title and the first 300 characters of the first chunk; they are not annotator
labels.

Counts used below (all reproduce the freeze):

| quantity | n |
|---|---|
| trace instances | 636 (EXACT 256, PARTIAL 46, ABSENT 334) |
| traced (EXACT+PARTIAL) | 302 |
| traced and present (§3) | 190 (EXACT 156, PARTIAL 34) |
| traced and absent | 112 (EXACT 100, PARTIAL 12), 103 distinct titles |
| distinct corpus pages behind the 190 | 182 |
| qids with ≥1 Xp page | 136; histogram {0:64, 1:93, 2:34, 3:7, 4:2} |

## 1. The trace rule (efbpt_prepare_stage0.py:243–290, quoted verbatim)

```python
def normalize_lexical(value: Any) -> str:
    lowered = str(value).lower()
    without_punctuation = re.sub(r"[^\w\s]", " ", lowered)
    return " ".join(without_punctuation.split()).strip()


def singularize(token: str) -> str:
    return token[:-1] if token.endswith("s") and len(token) > 3 else token


def token_equal(left: str, right: str) -> bool:
    return singularize(left) == singularize(right)


def classify_lexical_trace(question_en: str, gold_title: str) -> dict[str, Any]:
    normalized_question = normalize_lexical(question_en)
    normalized_title = normalize_lexical(gold_title)
    question_tokens = normalized_question.split()
    title_tokens = normalized_title.split()
    require(title_tokens, f"empty lexical title after normalization: {gold_title!r}")

    exact = any(
        all(token_equal(title_token, question_tokens[start + offset])
            for offset, title_token in enumerate(title_tokens))
        for start in range(len(question_tokens) - len(title_tokens) + 1)
    )
    if exact:
        trace_class = "EXACT"
        head = title_tokens[title_tokens.index("of") - 1] if "of" in title_tokens else title_tokens[-1]
    else:
        if "of" in title_tokens:
            of_index = title_tokens.index("of")
            require(of_index > 0, f"title has no token before first 'of': {gold_title!r}")
            head = title_tokens[of_index - 1]
        else:
            head = title_tokens[-1]
        trace_class = (
            "PARTIAL" if any(token_equal(head, token) for token in question_tokens) else "ABSENT"
        )
    return {
        "normalized_question_en": normalized_question,
        "lexical_normalized_gold_title": normalized_title,
        "question_tokens": question_tokens,
        "title_tokens": title_tokens,
        "title_head_token": head,
        "lexical_trace_class": trace_class,
        "lexical_trace_en": "YES" if trace_class in {"EXACT", "PARTIAL"} else "NO",
    }
```

PARTIAL in one sentence: the full lower-cased, punctuation-stripped title does
not occur as a contiguous token sequence in the question, but the title's
"head" token (the token before the first `of`, otherwise the last token)
equals some question token after crude singularisation.

Two properties of the rule matter for the audit. First, a parenthetical
disambiguator is not stripped, so for "Racket (sports equipment)" the head is
"equipment" and "Crane (bird)" has head "bird"; such titles can only be EXACT
if the parenthetical itself is in the question, which never happens, and they
fall to ABSENT even when the bare title is in the question (see §3). Second,
PARTIAL fires on any single shared head token, including common nouns
("year", "bee", "milk", "color", "dynasty"), so PARTIAL is a weak signal.

## 2. Wrong page included

### 2a. All 34 PARTIAL-and-present instances

(a) = is this page the right subject for the question; (b) = is the title
really named in the question rather than sharing one common word.

| # | qid | question_en | gold title = corpus title (rule) | (a) | (b) | reason for any "no" |
|---|---|---|---|---|---|---|
| P1 | 014307… | Can you buy spinal cord at Home Depot? | The Home Depot (exact) | yes | yes | — (PARTIAL only because the article "The" is missing) |
| P2 | 01c278… | Did Supernatural break 2001 CW debuting shows seasons record? | The CW (exact) | yes | yes | — |
| P3 | 06724d… | Could one Amazon share ever buy twenty year Netflix subscription? | Fiscal year | no | no | an accounting concept; the question needs Amazon's share price and Netflix's price; only "year" is shared |
| P4 | 1a4eb8… | Is Disney associated with Los Angeles County? | Central Los Angeles | no | no | a sub-region of the city, not Disney or the county; "Central" is not in the question |
| P5 | 1a4eb8… | same | Downtown Los Angeles | no | no | same as P4 |
| P6 | 24418d… | Is basil safe from Hypervitaminosis D? | Vitamin D | yes | no | relevant page, but the match is the single token "D" |
| P7 | 3b1834… | Would a CEO typically clean the toilets in a company's building? | Founder CEO | no | no | the question is about CEOs in general; only "CEO" shared |
| P8 | 3b38e0… | Does Mario use mushrooms to run faster? | Super Mario | yes | yes | — (named by its short name) |
| P9 | 3bba4e… | Will Oasis cruise boat traverse the Lincoln Tunnel? | Oasis of the Seas | yes | yes | — ("Oasis cruise boat") |
| P10 | 3fc11f… | Is Thanksgiving sometimes considered a day of mourning? | National Day of Mourning (United States protest) | yes | yes | — |
| P11 | 42c947… | Can oysters be used in guitar manufacturing? | Inlay (guitar) | yes | no | right evidence (mother-of-pearl inlay) but "inlay" is never named; only "guitar" |
| P12 | 48da75… | Was only woman to serve as U.S. Speaker of the House alive during the attack on Pearl Harbor? | Speaker of the United States House of Representatives | yes | yes | — |
| P13 | 4bbf4b… | Can you swim to Miami from New York? | Miami River (New York) | **no** | no | an upstate-New-York stream; the question means Miami, Florida. The gold evidence id itself points to the wrong page (annotation error inherited from StrategyQA) |
| P14 | 502d4d… | Is Home Depot a one stop shop for crucifixion supplies? | The Home Depot | yes | yes | — |
| P15 | 5ab66b… | Would it be common to find a penguin in Miami? | Chinstrap penguin | no | no | one species; the question is about penguins generally; only "penguin" shared |
| P16 | 73c521… | Does Lemon enhance the flavor of milk? | Soured milk | yes | no | the page carries the needed fact (lemon juice curdles milk) but "soured" is not named |
| P17 | 7dc74f… | Are queen bees unnecessary for growing apples? | Mason bee | yes | no | the needed fact (queenless pollinators) but only "bee" shared |
| P18 | 7f4eff… | Could an American confuse breakfast in British cuisine for dinner? | Full breakfast | yes | no | "full" not named; only "breakfast" |
| P19 | 8d3dda… | Did Brazilian jiu-jitsu Gracie founders have at least a baker's dozen of kids between them? | Carlos Gracie | yes | yes | — ("Gracie founders" is a proper-noun reference) |
| P20 | 8d3dda… | same | Hélio Gracie | yes | yes | — |
| P21 | 8e03ed… | Could all of the famous Apollo's hypothetically defeat all of the famous D'Artagnan's? | Charles de Batz de Castelmore d'Artagnan | yes | yes | — |
| P22 | 90fc9a… | Would Dante have hypothetically placed Nostradamus in 3rd Circle of Hell? | Inferno (Dante) | yes | no | the work is referenced by description ("Circle of Hell") and its author, not by its title |
| P23 | 959fb2… | Did Naruto escape the Temple of Doom? | Indiana Jones and the Temple of Doom | yes | yes | — |
| P24 | 9883d6… | Do onions have a form that resembles the inside of a tree? | Fried onion | no | no | the question is about onion layers/rings; a snack-food page; only "onion" |
| P25 | 9bb594… | Can you save every HD episode of Game of Thrones on Samsung Galaxy A10e? | Game of Thrones (season 1) | yes | yes | — (series named; season page holds episode counts) |
| P26 | 9bb594… | same | The Iron Throne (Game of Thrones) | no | no | the finale episode page; useful only for the total episode count; "Iron Throne" not named |
| P27 | a406e0… | Would the trees in Auburn, New York be changing colors in September? | Autumn leaf color | yes | no | only "color(s)" shared |
| P28 | b74793… | Did origin dynasty of Go precede Jia Sidao? | Song dynasty | yes | no | Jia Sidao's dynasty is the right subject, but "Song" is not named; only "dynasty" |
| P29 | cac8e6… | Is number of different US President's in 1800s a lucky number in Hong Kong? | USS President (1800) | **no** | no | a frigate; "President" and "1800" coincide with different meanings. Gold evidence id is wrong (annotation error) |
| P30 | e26a56… | Will Futurama surpass the number of episodes of The Simpsons by the end of 2020? | Meanwhile (Futurama) | no | no | the series finale page; it does carry the count (140 episodes) but the subject is Futurama; "Meanwhile" not named |
| P31 | e6d397… | Can Africanized bees be considered multicultural? | Honey bee | yes | no | only "bee(s)" |
| P32 | e6d397… | same | Western honey bee | yes | no | only "bee(s)" |
| P33 | f7b1cc… | Is breast cancer associated with a ribbon? | Awareness ribbon | yes | no | only "ribbon" |
| P34 | fdeec1… | Are you likely to find a crucifix in Karachi? | Religion in Karachi | yes | no | the city is named, the topic page is not |

PARTIAL-present totals: (a) right subject 24/34 (70.6%), wrong subject 10/34;
(b) title really named 12/34 (35.3%), one shared common word 22/34 (64.7%).
Two of the ten wrong-subject pages (P13, P29) are wrong at the gold-evidence
level, i.e. the StrategyQA annotation points to the wrong article. The other
eight are tangential or over-specific pages that the annotator used as a
stepping stone.

### 2b. 40 random EXACT-and-present instances (seed 20261027)

| # | qid | question_en | gold title = corpus title | (a) | (b) | note |
|---|---|---|---|---|---|---|
| E1 | 3bba4e… | Will Oasis cruise boat traverse the Lincoln Tunnel? | Lincoln Tunnel | yes | yes | |
| E2 | 0d2252… | Is a fairy more prevalent in world myths than a valkyrie? | Valkyrie | yes | yes | |
| E3 | 3486b6… | Do more Cauliflower grow in Arizona than California? | Arizona | yes | yes | |
| E4 | 48da75… | Was only woman to serve as U.S. Speaker of the House alive during the attack on Pearl Harbor? | Attack on Pearl Harbor | yes | yes | |
| E5 | f7a32d… | Can Herpes simplex virus spread on Venus? | Virus | yes | **no** | the question names "Herpes simplex virus"; the generic "Virus" page matches on one word inside that name (qid is one of the three excluded from D4-68) |
| E6 | a54b16… | Will a rock float in the atmosphere of Earth? | Atmosphere | yes | yes | |
| E7 | 25a088… | Can eating grapefruit kill besides allergies or choking? | Grapefruit | yes | yes | |
| E8 | 684d4c… | Do Christians anticipate an existence in Sheol after death? | Sheol | yes | yes | |
| E9 | 064d8f… | Would the crew of Apollo 15 have difficulty riding a unicycle? | Unicycle | yes | yes | |
| E10 | 06724d… | Could one Amazon share ever buy twenty year Netflix subscription? | Netflix | yes | yes | |
| E11 | d11421… | Did Gandhi watch the television show Bonanza? | Bonanza | yes | yes | |
| E12 | ae0891… | Would Paul Bunyan hypothetically be a poor choice for an urban planner? | Urban planner | yes | yes | |
| E13 | e87b63… | Is Michael Vick on People for the Ethical Treatment of Animals's hypothetical blacklist? | People for the Ethical Treatment of Animals | yes | yes | |
| E14 | e53068… | Could a cat ride Panzer VIII Maus tank missile from Barcelona to Madrid? | Panzer VIII Maus | yes | yes | |
| E15 | 076f90… | Is lunch on the beach a good activity to spot the full circle of a rainbow? | Rainbow | yes | yes | |
| E16 | 16c08e… | Was a USB flash drive used in The Godfather? | The Godfather | yes | yes | |
| E17 | b64be9… | Is the Easter Bunny popular in September? | Easter Bunny | yes | yes | |
| E18 | 690a36… | Did Jack Dempsey fight the current WBC heavyweight champion? | Jack Dempsey | yes | yes | |
| E19 | f07a6a… | Does Andrew Johnson's presidential number exceed Elagabalus's Emperor number? | Andrew Johnson | yes | yes | |
| E20 | 5897ec… | As of 2020 have more women succeeded John Key than preceded him? | John Key | yes | yes | |
| E21 | 73c521… | Does Lemon enhance the flavor of milk? | Lemon | yes | yes | |
| E22 | 9bb594… | Can you save every HD episode of Game of Thrones on Samsung Galaxy A10e? | Game of Thrones | yes | yes | |
| E23 | 42c947… | Can oysters be used in guitar manufacturing? | Oyster | yes | yes | |
| E24 | 4e1b65… | Is capturing giant squid in natural habitat impossible with no gear? | Giant squid | yes | yes | |
| E25 | d01453… | Can the United Nations Framework Convention on Climate Change be held at the Javits Center? | Javits Center | yes | yes | |
| E26 | 8d06b6… | Is the title of Shirley Bassey's 1971 diamond song a true statement? | Shirley Bassey | yes | yes | |
| E27 | f4a336… | Is a northern fur seal needing emergency surgery in July likely a safe anesthesia candidate? | Anesthesia | yes | yes | |
| E28 | f8e608… | Could ancient Tony Bennett have a baby in 2020? | Tony Bennett | yes | yes | |
| E29 | 73ca2e… | Was Darth Vader monogamous? | Darth Vader | yes | yes | **first chunk is raw wikitext** (`{{Infobox character | name = Darth Vader | image = …`), so 1 of the 3 chunks given to the extractor is markup, not prose |
| E30 | 481029… | Could ten gallons of seawater crush a six year old? | Seawater | yes | yes | |
| E31 | b8df6a… | Can bottlenose dolphins hypothetically outbreed human women? | Bottlenose dolphin | yes | yes | |
| E32 | c928aa… | Would alligator best saltwater crocodile in hypothetical Lake Urmia battle? | Lake Urmia | yes | yes | |
| E33 | 3cec99… | Is Y2K relevant to the plot of The Godfather? | The Godfather | yes | yes | |
| E34 | 0d2252… | Is a fairy more prevalent in world myths than a valkyrie? | Fairy | yes | yes | |
| E35 | 014307… | Can you buy spinal cord at Home Depot? | Spinal cord | yes | yes | |
| E36 | e87b63… | Is Michael Vick on People for the Ethical Treatment of Animals's hypothetical blacklist? | Michael Vick | yes | yes | |
| E37 | a0896d… | Is Atlantic cod found in a vegemite sandwich? | Atlantic cod | yes | yes | |
| E38 | 103255… | Are goats found on abyssal plains? | Goat | yes | yes | |
| E39 | 8d3dda… | Did Brazilian jiu-jitsu Gracie founders have at least a baker's dozen of kids between them? | Brazilian jiu-jitsu | yes | yes | |
| E40 | 267609… | Can the Swiss Guard fill the Virginia General Assembly chairs? | Swiss Guards | yes | yes | the question means the Pontifical Swiss Guard; the page covers it |

EXACT-sample totals: (a) 40/40 right subject; (b) 39/40 really named (E5 is a
one-word match inside a longer name). One page (E29) delivers a wikitext
infobox as its first chunk.

Combined over the 74 judged included instances: wrong subject 10/74 (13.5%),
all of them PARTIAL. Extrapolated to the 190 Xp pages (156 EXACT at about 0%
wrong, 34 PARTIAL at 29% wrong) roughly 10 of 190 included pages (about 5%)
are the wrong subject, and 22 of 190 (12%) are matched on one shared common
word even though the page happens to be relevant.

## 3. Right page left out: 40 random ABSENT-trace instances (seed 20261027)

"Named in another form" = synonym, abbreviation, plural, possessive, spelling,
or alias. The last column shows whether the page exists in the corpus under §3
(not asked, but it separates linker loss from corpus loss).

| # | qid | question_en | gold title (head) | named in another form | reason | in corpus |
|---|---|---|---|---|---|---|
| A1 | 3503e8… | Would a responsible bartender make a drink for Millie Bobby Brown? | Legal drinking age (age) | no | implied by "bartender", not named | yes |
| A2 | 1534ef… | Is a jellyfish safe from atherosclerosis? | Circulatory system (system) | no | implied by "atherosclerosis" | yes |
| A3 | 30fcf6… | Were the Great Pyramids built by a theocratic government? | Giza pyramid complex (complex) | yes | "the Great Pyramids" is a common synonym of the Giza complex | yes |
| A4 | 3d7050… | Did Melania Trump have same profession as Olga Kurylenko? | Olga Storozhenko (storozhenko) | no | a different person (a Miss Ukraine) from the Olga Kurylenko asked about; the gold evidence looks mislinked | yes |
| A5 | e12fef… | Can someone from New England profit by growing coffee? | Climate of New England (climate) | no | "New England" is named, the climate topic is not | yes |
| A6 | 3cec99… | Is Y2K relevant to the plot of The Godfather? | Year 2000 problem (problem) | yes | abbreviation "Y2K" | yes |
| A7 | 8cfde6… | Is it possible to get killed walking to the Very Large Telescope? | Evan Tanner (tanner) | no | a person who died in a desert; not named | yes |
| A8 | 265dd5… | Do the anchors on Rede Globo speak Chinese? | Brazil (brazil) | no | implied by Rede Globo | yes |
| A9 | 46428d… | Would Eric Clapton's mother hypothetically be unable to legally purchase cigarettes in the USA at his birth? | Tobacco 21 (21) | no | the law is not named | yes |
| A10 | 069c88… | Can a sea turtle play tennis using a tennis racket? | Hand (hand) | no | not named | yes |
| A11 | 069c88… | same | Racket (sports equipment) (equipment) | yes | "racket" is in the question verbatim; ABSENT only because the head token is taken from the parenthetical | yes |
| A12 | a54b16… | Will a rock float in the atmosphere of Earth? | Rock (geology) (geology) | yes | "rock" verbatim; parenthetical artefact | yes |
| A13 | 8e03ed… | Could all of the famous Apollo's hypothetically defeat all of the famous D'Artagnan's? | Apollo Creed (creed) | yes | first name "Apollo" (possessive/plural form) | yes |
| A14 | bfb0f6… | Did Solomon make up bigger percentage of Islamic prophets than Kings of Judah? | Zedekiah (zedekiah) | no | one of the kings, not named | yes |
| A15 | 076f90… | Is lunch on the beach a good activity to spot the full circle of a rainbow? | Halo (optical phenomenon) (phenomenon) | no | not named | yes |
| A16 | 1c5695… | Did mercenaries fight for England in the Glorious Revolution? | Wincanton Skirmish (skirmish) | no | not named | yes |
| A17 | 9247b5… | Can a minotaur hypothetically injure a tibia playing football? | Anterior cruciate ligament injury (injury) | no | "injure" is a verb, the ACL is not named | yes |
| A18 | 481029… | Could ten gallons of seawater crush a six year old? | Child (child) | yes | paraphrase "six year old" | yes |
| A19 | a54b16… | Will a rock float in the atmosphere of Earth? | Sandstone (sandstone) | no | not named | no |
| A20 | 3f8a1b… | Would a hedgehog avoid animals without a spinal cord? | Vertebrate (vertebrate) | no | described by negation ("without a spinal cord"), not named | no |
| A21 | 4e1b65… | Is capturing giant squid in natural habitat impossible with no gear? | Scuba diving (diving) | no | "gear" only | yes |
| A22 | 80aa76… | Are sables related to wolverines? | Gulo (gulo) | yes | "wolverines" is the common name of genus Gulo | no |
| A23 | 38ef97… | Is blonde hair green eyed Sara Paxton considered a Latino? | Mexico (mexico) | no | not named | yes |
| A24 | 3ebb62… | Do movies always show nerds as the losers? | Superbad (film) (film) | no | not named | no |
| A25 | a4de29… | If you're running focal fossa, are you using linux? | Ubuntu (ubuntu) | yes | "focal fossa" is the Ubuntu 20.04 codename (alias) | yes |
| A26 | 73ca2e… | Was Darth Vader monogamous? | Monogamy (monogamy) | yes | adjective form "monogamous" | yes |
| A27 | 8cfde6… | Is it possible to get killed walking to the Very Large Telescope? | Desert (desert) | no | not named | yes |
| A28 | 502d4d… | Is Home Depot a one stop shop for crucifixion supplies? | Descriptions in antiquity of the execution cross (antiquity) | no | not named | yes |
| A29 | 7ea958… | Would an Evander Holyfield 2020 boxing return set age record? | Steve Ward (boxer) (boxer) | no | the record holder is not named | yes |
| A30 | 728147… | Can crane slamdunk? | Crane (bird) (bird) | yes | "crane" verbatim; parenthetical artefact | yes |
| A31 | 20693e… | Do the Egyptian pyramids look the same from outside as they did when new? | Limestone (limestone) | no | not named | no |
| A32 | 01c278… | Did Supernatural break 2001 CW debuting shows seasons record? | The WB (wb) | no | predecessor network, not named | yes |
| A33 | 2bc94c… | Could Christopher Walken enlist in the United States Marine Corps? | United States Armed Forces (forces) | no | a component (Marine Corps) is named, the whole is not | no |
| A34 | 4125c5… | Would a retail associate envy the retailer's CEO's pay? | Minimum wage (wage) | no | not named | no |
| A35 | 5897ec… | As of 2020 have more women succeeded John Key than preceded him? | Jenny Shipley (shipley) | no | not named | yes |
| A36 | 690a36… | Did Jack Dempsey fight the current WBC heavyweight champion? | Deontay Wilder (wilder) | no | described ("current WBC heavyweight champion"), not named | yes |
| A37 | c67945… | Could a snowy owl survive in the Sonoran? | Climate of the Arctic (climate) | no | not named | yes |
| A38 | fdeec1… | Are you likely to find a crucifix in Karachi? | Muslims (muslims) | no | not named | no |
| A39 | ba04b5… | Is the Matrix a standalone movie? | The Matrix (franchise) (franchise) | yes | "the Matrix" verbatim; parenthetical artefact | yes |
| A40 | f43533… | Can you house a giant squid at Soldier Field? | Gridiron football (football) | no | not named | yes |

ABSENT-sample totals: named in another form 11/40 (27.5%); not named 29/40.
Of the 11: 4 are a pure rule artefact (parenthetical disambiguator supplies
the head token: A11, A12, A30, A39), 2 are aliases/abbreviations (Y2K, focal
fossa), 1 is a morphological variant (monogamous), 3 are synonyms or
paraphrases (Great Pyramids, wolverines/Gulo, six year old/child) and 1 is a
first-name reference (Apollo). Stripping parentheticals before tracing would
move about 10% of ABSENT instances to EXACT at no cost. 31 of the 40 ABSENT
pages exist in the corpus, so for this class the loss is the linker, not the
corpus. One instance (A4) is a gold-evidence mislink.

## 4. Missing from corpus: 112 traced-but-absent instances (103 distinct titles)

Rule: for each absent title T, corpus titles t with norm(t) ⊂ norm(T) or
norm(T) ⊂ norm(t), t ≠ T after norm(), the 5 shortest kept. Candidates only;
no judgement is implied by listing. Bold marks the few that plausibly are the
same subject under another name.

| gold title (§3 rule) | trace classes | containment candidates (≤5 shortest) |
|---|---|---|
| 1980 United States presidential election | EXACT | United; Presidential; United States; United States presidential election; 1980 United States presidential election in Utah |
| Acela Express | EXACT | Acela; **Acela Express (trainset)** |
| African bee | PARTIAL | Bee; African |
| Alexander Hamilton | EXACT | Hamilton; Alexander; PS Alexander Hamilton; Alexander Hamilton Jr.; USS Alexander Hamilton |
| Anne Boleyn | EXACT | Boleyn; Anne Boleyn (play); Anne Boleyn (TV series); Coronation of Anne Boleyn; Anne Boleyn (disambiguation) |
| Barbas (Charmed) | PARTIAL | — |
| Beach | EXACT | Beachy; Beacham; Beacher; Beachey; Beachin |
| Cactus (all-caps: CACTUS) | EXACT, EXACT | **CACTUS**; Cactus V; CactusCon; Melocactus; Rapicactus; USS Cactus |
| Camel | EXACT | Camela; Camelin; Camella; Camelon; Escamel |
| Celery | EXACT | Celery salt; Celery vase; Leaf celery; Wild celery; Water celery |
| Chlorophyll | EXACT | Chlorophyll c; Chlorophyll d; Chlorophyll f; Chlorophyllin; Chlorophyllum |
| Cholera | EXACT | Cholera pit; Cholera belt; Fowl cholera; Cholera riots; Cholera toxin |
| Cigarette | EXACT | 20 Cigarettes; CBD cigarette; Cigarette box; USS Cigarette; SG (cigarette) |
| Cotton | EXACT | Cottone; Scotton; Cottonia; Bt cotton; Cottonade |
| Crucifix | EXACT | The Crucifix; Crucifix toad; Altar crucifix; Padua Crucifix; Santo Crucifixo |
| Crucifixion | EXACT, EXACT | Mond Crucifixion; Crucifixion thorn; White Crucifixion; Crucifixion (song); Crucifixion plaque |
| Diamond | EXACT | Diamond D; Diamond T; Diamond X; Diamond's; Diamondog |
| Douglas Adams | EXACT | Adams; Douglas; Douglas Adams (engineer); Douglas Adams at the BBC; Douglas Adams (cricketer) |
| Dr. Seuss | EXACT | Dr. Seuss's ABC; Dr. Seuss Memorial; Dr. Seuss Goes to War; Dr. Seuss's Sleep Book; In Search of Dr. Seuss |
| Drummer | EXACT | Ex Drummer; DigiDrummer; Drummer Boy; The Drummer; Drummerworld |
| EastEnders | EXACT | EastEnders Live; EastEnders: E20; Willy (EastEnders); EastEnders in Paris; Fatboy (EastEnders) |
| Easter | EXACT | Beaster; Eastern; Feaster; Reaster; Oreaster |
| Egyptian pyramids | EXACT | Egyptian; **List of Egyptian pyramids**; List of finds in Egyptian pyramids |
| Eiffel Tower | EXACT | Tower; Under the Eiffel Tower; A View from Eiffel Tower; Eiffel Tower (Cedar Fair); The Man on the Eiffel Tower |
| Elagabalus | EXACT | Elagabalus (deity) (the god, not the emperor asked about) |
| Email | EXACT | OzEmail; Tremail; Demailly; Movemail; Xhemaili |
| England | EXACT | England A; Go England; Mr England; Ty England; Yan England |
| Eric Clapton | EXACT | Eric; Clapton; Eric Clapton (album); Eric Clapton at His Best; Eric Clapton videography |
| Ethics | EXACT | Bioethics; Geoethics; Cyberethics; Role ethics; Viper Ethics |
| Fantastic Four | EXACT | Fantastic; Fantastic Four 2099; Fantastic Four in film; Maximum Fantastic Four; The New Fantastic Four |
| Flag of the United States | EXACT | OF; The; United; The United; United States |
| Franklin D. Roosevelt | PARTIAL | D.; Franklin; Roosevelt; Franklin D. Roosevelt Jr.; USS Franklin D. Roosevelt |
| French Revolution | EXACT | REvolution; French Revolution of 1848; French Revolutionary Army; French Revolutionary Wars; The Great French Revolution |
| Funeral | EXACT | Funeral Sky; Old Funeral; Jazz funeral; Thai funeral; Wifisfuneral |
| Félix Potvin | PARTIAL | Potvin |
| George Fox | EXACT | George; George Foxcroft; Austen George Fox; George Fox-Selwyn; George Fox (album) |
| Giant panda | PARTIAL | Giant; Basi (giant panda); An An (giant panda); Giant Panda (group); Taotao (giant panda) |
| Glorious Revolution | EXACT | Glorious; REvolution; Glorious Revolution (Spain); The Glorious Revolution (EP); Glorious Revolution in Scotland |
| Gorilla | EXACT | IGorilla; Gorilla FC; Gorillaman; La gorilla; RedGorilla |
| Great Pyramid of Giza | PARTIAL, PARTIAL, EXACT | OF; Giza; GReAT |
| Guitar | EXACT | Guitart; Guitaret; Guitarra; Guitarro; Saguitar |
| Hair | EXACT | Hairy; Hairé; Khair; Phair; Shair |
| Heart | EXACT | Hearth; Hearts; Hearty; Heartz; ToHeart |
| Hermes | EXACT | Hermes8; Hermesh; Thermes; Hermesia; Orohermes |
| Honey | EXACT | Honeyd; Honeyz; Honeyri; Honeyst; Mahoney |
| Human | EXACT | Humann; Humanz; IHuman; Shuman; Thuman |
| Hypervitaminosis D | EXACT | — |
| Ice | EXACT | Tice; VICE; Vice; Zice; Žice |
| Integrated circuit | PARTIAL | 2.5D integrated circuit; Die (integrated circuit); Integrated circuit design; Integrated circuit layout; Linear integrated circuit |
| Jackson Pollock | EXACT | Pollock; Who the $&% Is Jackson Pollock?; Jackson Pollock: An American Saga |
| Jane Austen | EXACT | Jane; Jane Austen Centre; Jane Austen College; Becoming Jane Austen; The Real Jane Austen |
| Judo | EXACT | Judoc; Cejudo; Judogi; Judoon; Kaijudo |
| Karachi | EXACT | Karachiya; Karachi FM; Karachi Zoo; Karachi Time; Lake Karachi |
| King Arthur | EXACT | King; ARTHUR; Arthur; King Arthur's Gold; King Arthur's Hall |
| Koala | EXACT, EXACT | El Koala; KoalaPad; Tankoala; Perikoala; The Koala |
| Kurt Cobain | EXACT | Kurt; Church of Kurt Cobain; Suicide of Kurt Cobain; Who Killed Kurt Cobain?; The Songs of Kurt Cobain |
| L. Ron Hubbard | EXACT | Ron; L. Ron; Hubbard; L. Ron Hubbard House; Excalibur (L. Ron Hubbard) |
| LaVeyan Satanism | PARTIAL | — |
| Last rites | EXACT | Last Rites (band); Last Rites (album); Last Rites (1975 film); Last Rites (1988 film); Last Rites of the Dead |
| Liberal conservatism | PARTIAL | — |
| MF Doom | EXACT | MF; Doom; MF Doom discography; Doomsday (MF Doom song); MF Doom & Trunks Presents Unicron |
| Mahatma Gandhi | PARTIAL | Mahatma; Mahatma Gandhi Park; Mahatma Gandhi Setu; Mahatma Gandhi Series; Mahatma Gandhi Stadium |
| Michael Bloomberg | EXACT | Michael; Bloomberg; Mayoralty of Michael Bloomberg; Electoral history of Michael Bloomberg; Political positions of Michael Bloomberg |
| Micronesia | EXACT | Micronesian; Micronesians; Flax micronesia; Micronesia Mall; Micronesian Games |
| Minor League Baseball | EXACT | Minor; League; Baseball; Minor League Baseball rosters; Minden (minor league baseball) |
| Minotaur | EXACT | Minotaure; Minotauri; Minotaur V; Minotaur-C; Minotauria |
| Moose (all-caps: MOOSE) | EXACT | **MOOSE**; Vamoose; LawMoose; Moose TV; Moosejaw; Mooseman |
| Northern fur seal | EXACT, EXACT | Fur; Northern |
| Nostradamus | EXACT | Hey Nostradamus!; Nostradamus Effect; Nostradamus (album); The Nostradamus Kid; Chiren (Nostradamus) |
| One stop shop | EXACT | Shop; One Stop Shopping Centre |
| Othello | EXACT | Othello Bah; Peneothello; Othello Radou; Othello error; Othello Hunter |
| Oxygen | EXACT | Doxygen; Foxygen; Oxygene; OxygenOS; Toxygene |
| Paul Bunyan | EXACT | Bunyan; Paul Bunyan Land; Paul Bunyan Mall; Paul Bunyans Cabin; Paul Bunyans Stump |
| Penguin | EXACT | Penguinone; SS Penguin; Penguin Rep; Penguindrum; USS Penguin |
| Pilgrimage | EXACT | Dry Pilgrimage; The Pilgrimage; Maya pilgrimage; Werl pilgrimage; Wild Pilgrimage |
| Porsche | EXACT | Porsche 64; Porsche SE; Porsche 993; Porsche B32; Porsche C88 |
| Prime Minister of the United Kingdom | EXACT | OF; The; United; The United; Deputy Prime Minister of the United Kingdom |
| Prime minister | EXACT | Mr. Prime Minister; Our Prime Minister; Prime Minister & I; Prime Minister Abe; Prime Minister Lee |
| Prime number | EXACT | Prime number theory; List of prime numbers; Largest known prime number; Miho Komatsu 7 : prime number; The Solitude of Prime Numbers |
| Prophet | EXACT | Prophete; A Prophet; Prophet-5; Propheteer; R. Prophet |
| Quartz | EXACT | Quartzy; XQuartz; Quartz 2D; Quartzite; USS Quartz |
| Rede Globo | EXACT | Rede; Globo; List of 11 PM telenovelas of Rede Globo |
| Richard Dawkins | EXACT | Dawkins; Richard; Richard Dawkins Award; Views of Richard Dawkins; Richard Dawkins bibliography |
| Richard Wagner | EXACT | Richard; Richard Wagner (judge); Richard Wagner Memorial; Richard Wagner Monument; Richard Wagner Foundation |
| Shiva | EXACT | Shivam; Shivaji; Shivali; Shivani; Yeshiva |
| Spaghetti | EXACT | Spaghettim; SpaghettiOs; Spaghettieis; Spaghetti sort; Spaghetti worm |
| Television | EXACT | B-television; I-Television; Television X; U Television; So Television |
| The Matrix | EXACT, EXACT | The; Enter the Matrix; Neo (The Matrix); The Matrix Online; Zion (The Matrix) |
| The Simpsons | EXACT | The; The Simpsons Guy; Lego The Simpsons; The Simpsons Game; The Simpsons Ride |
| Tom Cruise | EXACT | Tom; Cruise; Being Tom Cruise; Tom Cruise Purple; Tom Cruise filmography |
| Tom and Jerry | EXACT | And; Tom; Tom and Jerry Tales; Nibbles (Tom and Jerry); Tom and Jerry: The Movie |
| Turtle | EXACT | TurtleBot; Turtle Wax; Turtle pie; Turtleford; WildTurtle |
| United Nations Framework Convention on Climate Change | EXACT | On; Change; United; Framework; Convention |
| Vegemite | EXACT | The Vegemite Tales |
| Venus (all-caps: VENUS) | EXACT | **VENUS**; Evenus; Advenus; Sevenus; Venus 5; Venustus |
| Week | EXACT, EXACT | Gweek; IWeek; Sweek; Tweek; Weeke |
| Winter | EXACT | Kwinter; Winters; Winter's; Winteria; Winterval |
| Wolfgang Amadeus Mozart | PARTIAL | Amadeus; Wolfgang; Death of Wolfgang Amadeus Mozart; Wolfgang Amadeus Mozart and Prague; Wolfgang Amadeus Mozart monument, Baku |
| Wolverine | EXACT | Wolverine 1; Wolverine 2; USS Wolverine; Wolverine Air; Wolverine Pass |
| Woodrow Wilson | EXACT, EXACT | Wilson; Woodrow; John Woodrow Wilson; Woodrow Wilson Hall; Woodrow Wilson Pace |
| Year | EXACT | Yeary; Year 8; Year 9; Yearby; Yearim |
| Yellow pages | EXACT | Yellow Pages Group; Export Yellow Pages; Global Yellow Pages; Valley Yellow Pages; Yellow Pages Limited |
| Yuri Gagarin | EXACT | Yuri; Yuri Gagarin (band); Yuri Gagarin Stadium; Death of Yuri Gagarin; Monument to Yuri Gagarin |

Reading of §4. Only a handful of the 103 titles have a plausible same-subject
page under another name: the 3 all-caps duplicates (CACTUS, MOOSE, VENUS;
whether those are the real articles or mis-cased copies needs a look at their
text), "Acela Express (trainset)", "List of Egyptian pyramids", and arguably
"Elagabalus (deity)" is a different subject. The great majority are plain
Wikipedia articles with no counterpart at all: Beach, Camel, Cotton, Diamond,
Hair, Heart, Human, Ice, Oxygen, Week, Year, Winter, England, Television,
Easter, Heart, Guitar, Penguin, Turtle, Eiffel Tower, French Revolution,
Alexander Hamilton, Jane Austen, Tom Cruise, The Simpsons, The Matrix. These are
among the most-linked pages in Wikipedia, so their absence is a property of
this corpus build (the corpus does hold oddities such as "The", "OF", "And",
"ARTHUR", "GReAT" and "REvolution" as titles). This supports the freeze's
replacement-corpus branch: of the 302 traced instances, 112 (37%) are lost at
the corpus, not at the linker, and no amount of title matching recovers them.

## 5. Wrong part of page

### 5a. Chunks per page, the 190 included pages (182 distinct)

| total chunks in corpus | 1–3 | 4–6 | 7–12 | 13–24 | 25–49 | 50+ |
|---|---|---|---|---|---|---|
| distinct pages | 4 | 14 | 26 | 37 | 60 | 41 |

Median 28 chunks per page (quartiles 13 and 48; max 130). Only 4 of 182 pages
fit entirely in the 3 chunks the extractor sees; 111 pages have 20 or more
chunks. Averaged over pages, the first 3 chunks are 18.7% of the page text.
Per instance (190) the histogram is the same shape (full per-count histogram
in the scratch output; identical bins ±1). All 182 pages return 3 chunks except
the 4 short ones (1, 2, 3 chunks).

### 5b. Gold-fact keywords in the first 3 chunks, D4-68

Rule (rough, lexical): content words of a gold fact are alphanumeric tokens of
length ≥4 minus a small stop-list, crudely singularised; the first 3 chunks of
all gold pages of the question are tokenised the same way; a fact is "covered"
if at least half of its content words occur there.

| quantity | value |
|---|---|
| gold facts (urbench_facts over 68 qids) | 178 |
| facts with ≥50% of keywords present | 152 (85.4%) |
| facts with 100% of keywords present | 44 (24.7%) |
| mean keyword fraction per fact | 71.1% |
| qids with every fact ≥50% covered | 47 / 68 |
| qids with no fact ≥50% covered | 2 / 68 |

So the leads carry most of the vocabulary of the gold facts, but only a quarter
of the facts are fully covered lexically, and by §5a the extractor sees less
than a fifth of a typical page. "Wrong part of page" is therefore a real but
secondary loss for D4-68, and it will be larger for pages where the needed
fact is deep in the article (e.g. P30 Meanwhile (Futurama), P26 The Iron
Throne, where the count sits in the lead by luck of the article type).

## 6. Partition limit

```
$ scontrol show partition q_intel_share_L20 | grep -i maxtime
   MaxNodes=UNLIMITED MaxTime=UNLIMITED MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED MaxCPUsPerSocket=UNLIMITED
```

No wall-time cap on the partition; the 04:00:00 in d8_reachable_ceiling.sbatch
is a job-level choice, not a partition constraint.

## Summary for the D8 reading

- Included pages (Xp's 190): about 5% are the wrong subject (10 of the 34
  PARTIAL instances; 0 of 40 sampled EXACT), 2 of them wrong at the gold
  evidence itself (Miami River (New York), USS President (1800)). About 12% are
  included on a single shared common word yet happen to be relevant.
- Excluded pages (334 ABSENT): roughly 28% are nameable from the question
  (11/40), and 4 of those 11 are a rule artefact (parenthetical disambiguator
  as head token). A trace that strips "( … )" before computing the head would
  be a cheap, declared improvement for the D9 linker; it does not change D8.
- Corpus: 112 of 302 traced instances (37%) have no page under §3, and for
  almost all of them there is no page under any name. The missing set is
  dominated by core Wikipedia articles, so this is a corpus-build defect, not
  a linking defect; the replacement-corpus branch of the freeze is the only
  remedy for it.
- Page depth: the extractor sees 3 chunks of a median 28-chunk page (18.7% of
  text on average); gold-fact vocabulary is nonetheless ≥50% present for 85%
  of D4-68 facts, fully present for 25%.
- One data-quality note for the extractor: at least one included page (Darth
  Vader) delivers a raw wikitext infobox as its first chunk.
