# D15 analysis pack (descriptive)

Written by `eval/error_analysis_tests/efbpt/d15_analysis.py` (MD5 `bdf92668957dbbabf5e895c1e0b6a53c`) after the readout `docs/handoff/D15_READOUT.md` (MD5 `ed81a9412af53de47f777717ec28dbef`). It decides nothing and attaches no wording; the readout is the record. PASS CHECK: the accuracy and the correct count of every arm on set A and on all sampled rows (22 values) recomputed here equal the readout.

b = first arm right and second wrong; p = exact two-sided McNemar. UrduMMLU rows are on set A (n = 2600 of 2600 sampled rows; 100 rows per subdomain, so the UrduMMLU accuracy is the mean over subjects, not the dataset-weighted accuracy).

## A. Reading effect over all five sets

First arm = the model with the English translation; second arm = the same model on the native item. StrategyQA, CSQA, PIQA and GSM8K: the 12 lines of `docs/handoff/D14_ANALYSIS.md` §A copied as they stand (their source column points into D13_ANALYSIS.md and D14_READOUT.md; StrategyQA eval458 with the D13 prompts; CSQA, PIQA, GSM8K all rows; these four Urdu sets were translated from English). UrduMMLU: native Urdu, set A, LA lines of `docs/handoff/D15_READOUT.md`. For T and M the D15 test (H1) compares with the English instruction and the Urdu item (`_ue`); the D14 form (against the Urdu prompt, `_ur`) is shown too.

| task | model | first arm vs second arm | n | acc first % | acc second % | delta pp | b | c | raw p | source line |
|---|---|---|---|---|---|---|---|---|---|---|
| StrategyQA | qwen3.8-max | API_tr_R1 vs API_direct_ur_eval | 458 | 80.13 | 67.47 | +12.66 | 79 | 21 | 4.337e-09 | D13_ANALYSIS.md:13 |
| StrategyQA | Llama-3.1-70B | T_tr_R1 vs T_direct_F_eval | 458 | 75.11 | 63.32 | +11.79 | 112 | 58 | 4.186e-05 | D13_ANALYSIS.md:15 |
| StrategyQA | Qwen3-14B | RO_eval vs A_r2_eval | 458 | 64.63 | 54.80 | +9.83 | 106 | 61 | 0.0006187 | D13_ANALYSIS.md:19 |
| CSQA | qwen3.8-max | M_r1 vs M_ur | 1500 | 80.33 | 75.80 | +4.53 | 137 | 69 | 2.49e-06 | D14_READOUT.md:127 |
| CSQA | Llama-3.1-70B | T_r1 vs T_ur | 1500 | 74.20 | 56.93 | +17.27 | 365 | 106 | 2.40e-34 | D14_READOUT.md:123 |
| CSQA | Qwen3-14B | S_ro vs S_ur | 1500 | 74.20 | 68.53 | +5.67 | 172 | 87 | 1.40e-07 | D14_READOUT.md:119 |
| PIQA | qwen3.8-max | M_r1 vs M_ur | 750 | 93.07 | 92.40 | +0.67 | 31 | 26 | 0.5966 | D14_READOUT.md:139 |
| PIQA | Llama-3.1-70B | T_r1 vs T_ur | 750 | 86.80 | 78.93 | +7.87 | 107 | 48 | 2.44e-06 | D14_READOUT.md:135 |
| PIQA | Qwen3-14B | S_ro vs S_ur | 750 | 77.33 | 71.60 | +5.73 | 114 | 71 | 0.0019 | D14_READOUT.md:131 |
| GSM8K | qwen3.8-max | M_r1 vs M_ur | 700 | 94.43 | 94.57 | -0.14 | 8 | 9 | 1.0000 | D14_READOUT.md:151 |
| GSM8K | Llama-3.1-70B | T_r1 vs T_ur | 700 | 94.43 | 86.14 | +8.29 | 77 | 19 | 1.86e-09 | D14_READOUT.md:147 |
| GSM8K | Qwen3-14B | S_ro vs S_ur | 700 | 91.14 | 87.43 | +3.71 | 44 | 18 | 0.0013 | D14_READOUT.md:143 |
| UrduMMLU | qwen3.8-max | M_r1 vs M_ue | 2600 | 87.46 | 89.15 | -1.69 | 83 | 127 | 0.0029 | D15_READOUT.md:125 |
| UrduMMLU | Llama-3.1-70B | T_r1 vs T_ue | 2600 | 80.69 | 67.46 | +13.23 | 504 | 160 | 2.38e-42 | D15_READOUT.md:121 |
| UrduMMLU | Qwen3-14B | S_ro vs S_ur | 2600 | 77.38 | 72.85 | +4.54 | 256 | 138 | 2.89e-09 | D15_READOUT.md:117 |
| UrduMMLU | qwen3.8-max | M_r1 vs M_ur (D14 form) | 2600 | 87.46 | 88.77 | -1.31 | 91 | 125 | 0.0245 | D15_READOUT.md:263 |
| UrduMMLU | Llama-3.1-70B | T_r1 vs T_ur (D14 form) | 2600 | 80.69 | 65.15 | +15.54 | 555 | 151 | 4.19e-55 | D15_READOUT.md:259 |

## B. Subdomains by S_ro − S_ur (set A)

Cells copied from `docs/handoff/D15_READOUT.md` §10 (delta on all rows of the subdomain in set A; about 100 rows each, so one subdomain delta moves by 1 pp per row). Sorted here; ties by name.

### 10 largest

| subdomain | group | domain | n (set A) | S_ur % | S_ro − S_ur | T_r1 − T_ue | M_r1 − M_ue | S_self − S_ur | T_self − T_ue | S_tr − S_ur |
|---|---|---|---|---|---|---|---|---|---|---|
| economics | OTHER | Social Sciences | 100 | 75.00 | +13.00 | +21.00 | +1.00 | +5.00 | -18.00 | +11.00 |
| geography | OTHER | Social Sciences | 100 | 74.00 | +12.00 | +14.00 | +1.00 | +6.00 | -10.00 | +10.00 |
| fine arts | OTHER | Humanities | 100 | 62.00 | +11.00 | +16.00 | +2.00 | -5.00 | -6.00 | -1.00 |
| chemistry | OTHER | STEM | 100 | 86.00 | +10.00 | +26.00 | -2.00 | +3.00 | -3.00 | +8.00 |
| professional studies | OTHER | Profession | 100 | 82.00 | +9.00 | +10.00 | -5.00 | +1.00 | -1.00 | +5.00 |
| urdu grammar | URDU | Humanities | 100 | 47.00 | +9.00 | +7.00 | -8.00 | -10.00 | -16.00 | -1.00 |
| health and physical education | OTHER | Social Sciences | 100 | 69.00 | +8.00 | +19.00 | -5.00 | +1.00 | -16.00 | +4.00 |
| computer science | OTHER | STEM | 100 | 89.00 | +7.00 | +17.00 | +0.00 | +6.00 | -22.00 | +3.00 |
| ethics | OTHER | Humanities | 100 | 72.00 | +7.00 | +11.00 | -1.00 | +0.00 | -5.00 | +12.00 |
| general knowledge | OTHER | Other | 100 | 69.00 | +6.00 | +4.00 | -1.00 | +4.00 | +1.00 | +4.00 |

### 10 smallest

| subdomain | group | domain | n (set A) | S_ur % | S_ro − S_ur | T_r1 − T_ue | M_r1 − M_ue | S_self − S_ur | T_self − T_ue | S_tr − S_ur |
|---|---|---|---|---|---|---|---|---|---|---|
| pakistan studies | REGIONAL | Social Sciences | 100 | 81.00 | -8.00 | +9.00 | -2.00 | -6.00 | -12.00 | -1.00 |
| urdu language | URDU | Humanities | 100 | 50.00 | -4.00 | +8.00 | -6.00 | -3.00 | -12.00 | +9.00 |
| current and international affairs | OTHER | Social Sciences | 100 | 46.00 | -4.00 | -3.00 | +2.00 | -5.00 | -6.00 | -10.00 |
| psychometrics | OTHER | Social Sciences | 100 | 42.00 | +1.00 | +10.00 | +3.00 | +4.00 | +0.00 | -5.00 |
| mathematics | OTHER | STEM | 100 | 95.00 | +1.00 | +15.00 | +0.00 | +0.00 | -55.00 | +1.00 |
| islamic studies | REGIONAL | Humanities | 100 | 73.00 | +1.00 | +9.00 | -8.00 | -4.00 | -8.00 | -2.00 |
| education | OTHER | Social Sciences | 100 | 72.00 | +2.00 | +12.00 | +0.00 | +0.00 | -8.00 | +5.00 |
| urdu literature | URDU | Humanities | 100 | 32.00 | +3.00 | -5.00 | -12.00 | -12.00 | -12.00 | +3.00 |
| physics | OTHER | STEM | 100 | 93.00 | +3.00 | +18.00 | +1.00 | +4.00 | -22.00 | +1.00 |
| commerce | OTHER | Social Sciences | 100 | 79.00 | +3.00 | +12.00 | +0.00 | -4.00 | -11.00 | -3.00 |

## C. Rows where the reading changes the answer of Qwen3-14B (set A)

Per group, 15 random rows (a new `random.Random(1515)` for each list; candidates in run order) where S_ro is right and S_ur wrong, and 15 where S_ro is wrong and S_ur right (all of them if fewer). Item = the Urdu item block; R1 = the R1 English translation; answers = the predictions read by the readout's scorer (empty = unparsed).

### URDU — S_ro right, S_ur wrong (40 candidates, 15 shown)

| row | subdomain | Urdu item | R1 translation | gold | S_ro | S_ur |
|---|---|---|---|---|---|---|
| um_13125 | urdu literature | Question: غزل میں اشعار کی کم از کم تعداد کتنی ہوتی ہے؟ / A: سات / B: نو / C: پانچ / D: تین | Question: What is the minimum number of couplets in a ghazal? / A: Seven / B: Nine / C: Five / D: Three | C | C | D |
| um_14900 | urdu literature | Question: ذوقی المعروف اور بحرالعرفان کہاں کے رہنے والے تھے / A: ان میں سے کوئی نہیں / B: دہلی / C: لکھنئو / D: لاہور | Question: Where did Zauqi Maroof and Behar-ul-Irfan belong to? / A: None of these / B: Delhi / C: Lucknow / D: Lahore | C | C | B |
| um_17767 | urdu literature | Question: کرشن چندر کس ادبی تحریک سے وابستہ تھے / A: ترقی پسند تحریک / B: حلقہ ارباب ذوق / C: ان میں سے کوئی نہیں / D: اسلامی تحریک | Question: Which literary movement was Krishan Chander associated with? / A: Progressive Movement / B: Halqa-e-Arbab-e-Zauq / C: None of these / D: Islamic Movement | A | A | B |
| um_18139 | urdu literature | Question: ہتیا اور دوسرے افسانے کے مصنف ہیں / A: آل احمد سرور / B: مجنوں گورکھ پوری / C: فراق گورکھ پوری / D: ان میں سے کوئی نہیں | Question: The author of Hatya aur Doosray Afsanay is / A: Ale Ahmad Suroor / B: Majnun Gorakhpuri / C: Firaq Gorakhpuri / D: None of these | B | B | C |
| um_19228 | urdu language | Question: وما ارسلنک الا رحمۃ اللعالمین، کا ترجمہ کیا ہے؟ / A: ہم نے آپ کو تمام مومنوں کیلئے رحمت بنا کے بھیجا۔ / B: ہم نے آپکو تمام جہانوں کیلئے رحمت بنا کے بھیجا۔ / C: ہم نے آپ کو تمام مسلمانوں کیلئے رحمت بنا کے بھیجا۔ / D: ہم نے آپ کو تمام کفار کیلئے ہدایت  ... | Question: What is the translation of "Wa ma arsalnaka illa rahmatan lil-alamin"? / A: We have sent you as a mercy for all believers. / B: We have sent you as a mercy for all the worlds. / C: We have sent you as a mercy for all Muslims. / D: We have sent you as ... | B | B | (unparsed) |
| um_19280 | urdu language | Question: مددگار میں "گار" کیا ہے؟ / A: ردیف / B: سابقہ / C: لاحقہ / D: قافیہ | Question: In "madadgar", what is "gar"? / A: Radif / B: Prefix / C: Suffix / D: Qaafiya | C | C | A |
| um_23031 | urdu grammar | Question: لیت و لعل کرناسے مراد ہے ؟ / A: ان میں سے کوئی نہیں / B: مالدار ہونا / C: ٹال مٹول کرنا / D: منت سماجت کرنا | Question: What is meant by "lait-o-la'al karna"? / A: None of these / B: To be wealthy / C: To procrastinate / to delay / D: To plead and beseech | C | C | A |
| um_23075 | urdu grammar | Question: بانس پر چڑھانا سے مراد ہے ؟ / A: ان میں سے کوئی نہیں / B: دراز قد / C: بدنام کرنا،تعریف کرنا / D: بلند آواز | Question: What does "Baans par charhana" mean? / A: None of these / B: Tall statured / C: To defame, to praise / D: Loud voice | C | C | A |
| um_23100 | urdu grammar | Question: پانچوں گھی میں ہونا سے مراد ہے ؟ / A: ان میں سے کوئی نہیں / B: بہت زیادہ گھی / C: مصیبت میں پھنس جانا / D: بہت فائدے میں ہونا | Question: What is meant by "panchon ghee mein hona"? / A: None of these / B: A lot of ghee / C: To get trapped in trouble / D: To be in great profit | D | D | A |
| um_23124 | urdu grammar | Question: تھوتھا چنا باجے گھنا سے مراد ہے؟ / A: کم ظرف بڑے دعوے کرنا / B: چنے کے کھیت / C: بانسری کی آواز / D: ان میں سے کوئی نہیں | Question: What is meant by "Thotha chana baaje ghana"? / A: A person of little substance making big claims / B: Chickpea field / C: Sound of the flute / D: None of these | A | A | C |
| um_23330 | urdu grammar | Question: خاکہ اڑانا سے مراد ہے ؟ / A: نقشا کھینچنا / B: ذلیل کرنا / C: تصویر بنانا / D: ان میں سے کوئی نہیں | Question: What does "khaka udana" mean? / A: To draw a sketch / B: To humiliate / C: To make a picture / D: None of these | B | B | D |
| um_23335 | urdu grammar | Question: انگشت بدنداں ہونا سے مراد ہے ؟ / A: ان میں سے کوئی نہیں / B: حیرت سے دانتوں میں انگلی دبانا / C: دانت میں درد ہونا / D: دانتوں سے ناخن کاٹنا | Question: What does "Angusht badandan hona" mean? / A: None of these / B: Biting one's finger in astonishment / C: Having a toothache / D: Cutting nails with teeth | B | B | A |
| um_23478 | urdu grammar | Question: جب کسی لفظ کو حقیقی معنوں کے بجائے مجازی معنوں میں اصل استعمال کیا جائے کہ حقیقی اور مجازی معنوں میں تشبیہہ کا تعلق موجود ہو تو اسے گرائمر کی رو سے کیا کہا جائے گا؟ / A: تشبیہہ تام / B: تشبیہہ مرکب / C: استعارہ / D: مستعارہ | Question: When a word is originally used in figurative meanings instead of literal meanings such that there exists a relationship of simile between the literal and figurative meanings, what will it be called according to grammar? / A: Tashbeeh-e-Taam / B: Tash ... | C | C | A |
| um_23507 | urdu grammar | Question: وہ تین لیٹر دودھ ال یا یی اسم صفت کی کونسی قسم ہے / A: صفت ذاتی / B: صفت نسبتی / C: صفت مقداری / D: سدت عددی | Question: Which type of adjective is "three liters of milk" — A or E? / A: Adjective of quality / B: Adjective of relation / C: Adjective of quantity / D: Numeral adjective | C | C | D |
| um_23584 | urdu grammar | Question: لفظ گال کا مترادف بتائیں؟ / A: چہرہ / B: ناتھا / C: ان میں سے کوئی بھی نہیں / D: رخسار | Question: Tell the synonym of the word "gaal" (cheek)? / A: Face / B: Natha / C: None of these / D: Cheek | D | D | C |

### URDU — S_ro wrong, S_ur right (32 candidates, 15 shown)

| row | subdomain | Urdu item | R1 translation | gold | S_ro | S_ur |
|---|---|---|---|---|---|---|
| um_13585 | urdu literature | Question: داستان امیر حمزہ کس کی کتاب ہے / A: بہادر علی حسینی / B: ان میں سے کوئی نہیں / C: خلیل خان اشک / D: کاظم علی جواں | Question: Whose book is Dastan-e-Amir Hamza / A: Bahadur Ali Hussaini / B: None of these / C: Khalil Khan Ashk / D: Kazim Ali Jawan | C | B | C |
| um_16780 | urdu literature | Question: تلخابہ شیریں کی صنف ہے / A: افسانہ / B: ان میں سے کوئی نہیں / C: شاعری / D: ناول | Question: The genre of Talkhaba Sheereen is / A: Short story / B: None of these / C: Poetry / D: Novel | C | A | C |
| um_19144 | urdu language | Question: سر وادی سینا کس کا مجموعہ کلام ہے؟ / A: احمد نواز / B: علامہ اقبال / C: منیر نیازی / D: فیض احمد فیض | Question: Sar-e-Wadi-e-Sina is a collection of poetry by whom? / A: Ahmad Nawaz / B: Allama Iqbal / C: Munir Niazi / D: Faiz Ahmad Faiz | D | A | D |
| um_19235 | urdu language | Question: جملہ درست کریں؟ کسی کو گالیاں مت نکالو / A: کسی کو گالیاں نہیں نکال / B: کسی کو گالیاں نہ نکالو / C: کسی کو گالیاں مت دو / D: کسی کو گالیاں مت نکال | Question: Correct the sentence? Do not hurl abuses at anyone / A: Do not hurl abuses at anyone / B: Do not hurl abuses at anyone / C: Do not give abuses to anyone / D: Do not hurl abuses at anyone | C | D | C |
| um_19536 | urdu language | Question: "نگری نگری پھرا مسا فر ‘‘ ناول کس نے لکھا ہے؟ " / A: خدیجہ مستور / B: نثار عزیز / C: جمیلہ ہاشمی / D: پروین شاکر | Question: "Who wrote the novel 'Nagri Nagri Phira Musafir'?" / A: Khadija Mastoor / B: Nisar Aziz / C: Jamila Hashmi / D: Parveen Shakir | B | A | B |
| um_19600 | urdu language | Question: ؟تین مصروں کی مختصر نظم کو کیا کہتے ہیں / A: مسمط / B: سانیٹ / C: ہائیکو / D: مربع | Question: What is a short poem of three stanzas called? / A: Musammat / B: Sonnet / C: Haiku / D: Murabba | C | A | C |
| um_20515 | urdu language | Question: بال بیکا کرنا محاورہ ہے۔ اس کا مطلب کیا ہے؟ / A: فائدہ پہنچانا / B: بال کی کھال اتارنا / C: کٹربیونت / D: نقصان پہنچانا | Question: "Baal beka karna" is an idiom. What does it mean? / A: To benefit / B: To split a hair / C: Nitpicking / D: To harm | D | C | D |
| um_20810 | urdu language | Question: لوحِ ایام کس کی تخلیق ہے؟ / A: ان میں سے کوئی نہیں / B: سمیع اللہ / C: مختار مسعود / D: وحیدہ نسیم | Question: Whose creation is "Lauh-e-Ayyam"? / A: None of these / B: Samiullah / C: Mukhtar Masood / D: Waheeda Naseem | C | A | C |
| um_21052 | urdu language | Question: ایسی نثرکا اصلاح میں کیا نام ہے جس کے دو فقروں کے تمام الفاظ ایک دوسرے کے ہم وزن اور حرف آخر بھی مواقف ہوں / A: مسجع نثر / B: مرصع / C: مرجز / D: مقفی | Question: What is the term in rhetoric for prose in which all the words of two sentences are of the same meter as each other and their final letters also match? / A: Musajja prose / B: Murassa / C: Murajjaz / D: Muqaffa | A | D | A |
| um_21414 | urdu language | Question: معنی کا انتحاب کریں : چکاچوند / A: گمراہ / B: روشنیاں / C: اندھیرا / D: ڈوب جانا | Question: Choose the meaning: Chakachond / A: Misguided / B: Lights / C: Darkness / D: To drown | B | A | B |
| um_22964 | urdu grammar | Question: گیدڑ بھبکی دیناسے مراد ہے ؟ / A: آوارہ گردی کرنا / B: خوشامد کرنا / C: فضول دھمکیاں دینا / D: ان میں سے کوئی نہیں | Question: What does "giddar bhabki dena" mean? / A: To wander around aimlessly / B: To flatter / C: To give empty threats / D: None of these | C | D | C |
| um_23052 | urdu grammar | Question: کایا پلٹ دینا سے مراد ہے ؟ / A: بہت پڑھا لکھا ہونا / B: کسی کو ملازم رکھنا / C: حالات بدلنا / D: ان میں سے کوئی نہیں | Question: What does "kaya palat dena" mean? / A: To be very educated / B: To employ someone / C: To change the circumstances / D: None of these | C | D | C |
| um_23079 | urdu grammar | Question: قلم توڑ دینا سے مراد ہے ؟ / A: فسادی ہونا / B: لاجواب نظم یا نثر لکھنا / C: کسی کو پھانسی دینا / D: ان میں سے کوئی نہیں | Question: What does "qalam tor dena" mean? / A: To be a troublemaker / B: To write an incomparable poem or prose / C: To hang someone / D: None of these | B | D | B |
| um_23443 | urdu grammar | Question: غزل اور قصیدے میں کس کا ہونا ضروری ہے؟ / A: مطلع / B: ردیف / C: قافیہ / D: مقطع | Question: What is essential in both a ghazal and a qasida? / A: Matla / B: Radif / C: Qaafiya / D: Maqta | C | A | C |
| um_23621 | urdu grammar | Question: ____کا پھل میٹھاہے / A: دکان / B: گھر / C: باغ / D: صبر | Question: The fruit of ____ is sweet / A: Shop / B: House / C: Garden / D: Patience | D | C | D |

### REGIONAL — S_ro right, S_ur wrong (16 candidates, 15 shown)

| row | subdomain | Urdu item | R1 translation | gold | S_ro | S_ur |
|---|---|---|---|---|---|---|
| um_05130 | pakistan studies | Question: تقسیم کے وقت درج ذیل میں سے کون سے حصوں کا غیر منصفانہ طور پر بھارت کے ساتھ الحاق کر دیا گیا؟ / A: گورداس پور، پٹھان کوٹ / B: مردان، کوہاٹ / C: سکھر، نواب شاہ / D: فیصل آباد، سرگودھا | Question: At the time of partition, which of the following areas were unjustly merged with India? / A: Gurdaspur, Pathankot / B: Mardan, Kohat / C: Sukkur, Nawabshah / D: Faisalabad, Sargodha | A | A | D |
| um_05870 | pakistan studies | Question: محمد یوسف کا تعلق کس کھیل سے تھا : / A: بیڈمنٹن / B: ہاکی / C: سنوکر / D: فٹبال | Question: Which sport was Muhammad Yousuf associated with: / A: Badminton / B: Hockey / C: Snooker / D: Football | C | C | B |
| um_06138 | pakistan studies | Question: پاکستان اور بھارت کی سرحد کی لمبائی کتنی ہے؟ / A: 4500 کلومیٹر / B: 3323 کلومیٹر / C: 2300 کلومیٹر / D: 3300 کلومیٹر | Question: What is the length of the border between Pakistan and India? / A: 4500 kilometers / B: 3323 kilometers / C: 2300 kilometers / D: 3300 kilometers | B | B | D |
| um_06383 | pakistan studies | Question: پاکستان کا قومی درخت کونسا ہے؟ / A: شیشم / B: پیپل / C: دیودار / D: نیم | Question: What is the national tree of Pakistan? / A: Sheesham / B: Peepal / C: Deodar / D: Neem | C | C | A |
| um_21479 | islamic studies | Question: قبرس کا جزیرہ دوبارہ کس نے فتح کیا؟ / A: السفاح / B: منصور / C: مامون الرشید / D: ہارون الرشید | Question: Who reconquered the island of Cyprus? / A: As-Saffah / B: Mansur / C: Mamun al-Rashid / D: Harun al-Rashid | D | D | C |
| um_21546 | islamic studies | Question: کراماً کاتبین کی تعداد ____ ہے / A: دو / B: چار / C: پانچ / D: تین | Question: The number of Kiraman Katibin is ____ / A: Two / B: Four / C: Five / D: Three | A | A | D |
| um_21606 | islamic studies | Question: قسم توڑنے پر ____ روزے لازم ہیں۔ / A: 4 / B: 5 / C: 2 / D: 3 | Question: For breaking an oath, ____ fasts are obligatory. / A: 4 / B: 5 / C: 2 / D: 3 | D | D | A |
| um_21839 | islamic studies | Question: دارارقم قائم ہوا : / A: ریاض / B: مدینہ / C: مکہ / D: طائف | Question: Dar al-Arqam was established: / A: Riyadh / B: Medina / C: Mecca / D: Taif | C | C | D |
| um_22006 | islamic studies | Question: مقسطون کے معنی کیا ہیں؟ / A: ان میں سے کوئی نہیں / B: عزت کرنا / C: قسط دینا / D: انصاف کرنے والے | Question: What is the meaning of Muqsitun? / A: None of these / B: To honor / C: To pay in installments / D: Those who do justice | D | D | A |
| um_22009 | islamic studies | Question: اعتصام کس کا صیغہ ہے؟ / A: ماضی کا / B: ان میں سے کوئی نہیں / C: امر کا / D: مصدر کا | Question: What form is I'tisam? / A: Past tense / B: None of these / C: Imperative / D: Infinitive | D | D | B |
| um_22108 | islamic studies | Question: نبی کریم صہ نے صلح حدیبیہ کے بعد جن فرماں رواؤں کو خط لکھا، ان میں مصر کے بادشاہ کا نام کیا تھا؟ / A: قیصر / B: مقوقس / C: کسریٰ / D: ان میں سے کوئی نہیں | Question: What was the name of the king of Egypt among the rulers to whom the Holy Prophet (PBUH) wrote letters after the Treaty of Hudaybiyyah? / A: Caesar / B: Muqawqis / C: Khosrow / D: None of these | B | B | A |
| um_22238 | islamic studies | Question: کان زکریا علیہ السلام ____؟ / A: حدادا / B: فلاحا / C: نجارا / D: ان میں سے کوئی نہیں | Question: Zakariya (peace be upon him) was a ____? / A: Blacksmith / B: Farmer / C: Carpenter / D: None of these | C | C | D |
| um_22557 | islamic studies | Question: میری قوم میں کسی غلطی پر کبھی اتفاق رائےنہیں ہو گا یہ کس کے الفاظ ہیں / A: حضرت علی / B: حضرت عمر / C: حضرت ابو بکر / D: حضرت محمد | Question: "My nation will never agree on an error" whose words are these / A: Hazrat Ali / B: Hazrat Umar / C: Hazrat Abu Bakr / D: Hazrat Muhammad | D | D | A |
| um_22703 | islamic studies | Question: حضرت عثمان رَضِیَ اللہُ تَعَالٰی عَنْہُ کی کون سی خوبی تاریخ اسلام کا ایک روشن باب ہے۔ / A: مالداری / B: فیاضی / C: بہادری / D: علم | Question: Which quality of Hazrat Usman RADI ALLAHU TA'ALA ANHU is a bright chapter in the history of Islam. / A: Wealth / B: Generosity / C: Bravery / D: Knowledge | B | B | D |
| um_22810 | islamic studies | Question: فاتح فارس کس صحابی کو کھا جاتا ھے؟ / A: عمر بن العاص / B: امیر معاویہ / C: سعد بن ابی واقاص / D: خالد بن ولید | Question: Who is called the conqueror of Persia? / A: Umar bin Al-Aas / B: Amir Muawiyah / C: Saad bin Abi Waqqas / D: Khalid bin Walid | C | C | D |

### REGIONAL — S_ro wrong, S_ur right (23 candidates, 15 shown)

| row | subdomain | Urdu item | R1 translation | gold | S_ro | S_ur |
|---|---|---|---|---|---|---|
| um_05200 | pakistan studies | Question: میر چاکر کس زبان کا مشہور ادیب ہے؟ / A: پنجابی / B: سندھی / C: بلوچی / D: پشتو | Question: Mir Chakar is a famous writer of which language? / A: Punjabi / B: Sindhi / C: Balochi / D: Pashto | C | A | C |
| um_05490 | pakistan studies | Question: پاکستان قومی اتحاد میں شامل سیاسی جماعتوں کی تعداد تھی۔ / A: 9 / B: 11 / C: 7 / D: 5 | Question: The number of political parties included in the Pakistan National Alliance was. / A: 9 / B: 11 / C: 7 / D: 5 | A | B | A |
| um_06162 | pakistan studies | Question: پاکستان کا دُوسرا نشانِ حیدر کس کو دیا گیا؟ / A: راشد منہاس / B: میجر عزیز بھٹی / C: کیپٹن راجہ محمد سرور / D: میجر محمد طفیل | Question: Who was awarded Pakistan's second Nishan-e-Haider? / A: Rashid Minhas / B: Major Aziz Bhatti / C: Captain Raja Muhammad Sarwar / D: Major Muhammad Tufail | D | A | D |
| um_06284 | pakistan studies | Question: خطبہ الہ آباد کب دیا گیا ؟ / A: 1930،دسمبر 19 / B: 1930،دسمبر 29 / C: 1933،دسمبر 29 / D: 1931،دسمبر 29 | Question: When was the Allahabad Address delivered? / A: 1930, December 19 / B: 1930, December 29 / C: 1933, December 29 / D: 1931, December 29 | B | D | B |
| um_06316 | pakistan studies | Question: خطبہ الہ آباد میں ۔۔۔۔نے ایک مسلمان ریاست کا تصور پیش کیا۔ / A: قائد اعظم / B: علامہ محمد اقبال / C: سرسید احمد خان / D: چوہدری رحمت علی | Question: In the Allahabad Address, .... presented the concept of a Muslim state. / A: Quaid-e-Azam / B: Allama Muhammad Iqbal / C: Sir Syed Ahmad Khan / D: Chaudhry Rahmat Ali | B | A | B |
| um_06434 | pakistan studies | Question: آئین کی کس ترمیم کی رو سے خیبر پختونخواہ میں فاٹا کو زم کیا گیا؟ / A: بائیس ویں ترمیم / B: بیسویں ترمیم / C: اٹھارھویں ترمیم / D: پچیسویں ترمیم | Question: Under which amendment to the Constitution was FATA merged into Khyber Pakhtunkhwa? / A: Twenty-second Amendment / B: Twentieth Amendment / C: Eighteenth Amendment / D: Twenty-fifth Amendment | D | C | D |
| um_06650 | pakistan studies | Question: قرارد داد پاکستان 1940 میں کس نے پیش کی ؟ / A: قائد اعظم / B: چوہدری رحمت علی / C: لیاقت علی خان / D: اے کے فضل الحق | Question: Who presented the Pakistan Resolution in 1940? / A: Quaid-e-Azam / B: Chaudhry Rahmat Ali / C: Liaquat Ali Khan / D: A. K. Fazlul Huq | D | A | D |
| um_06864 | pakistan studies | Question: پاکستان نے 1992 ء میں کس کی حکومت کو تسلیم کیا اور ان کے ساتھ سفارتی تعلقات قائم کیے؟ / A: افغان مجاہدین / B: شاہ ایران / C: شاہ ترکی / D: ان میں سے کوئی نہیں | Question: Whose government did Pakistan recognize in 1992 and establish diplomatic relations with? / A: Afghan Mujahideen / B: Shah of Iran / C: Shah of Turkey / D: None of these | A | D | A |
| um_21954 | islamic studies | Question: منکر سے مراد کیا ہے؟ / A: اچھائی / B: برائی / C: نیکی / D: ان میں سے کوئی نہیں | Question: What is meant by Munkar? / A: Goodness / B: Evil / C: Virtue / D: None of these | B | D | B |
| um_22134 | islamic studies | Question: حضرت ابو بکر صدیق رضہ نے کس کے ذریعے سے حضرت عائشہ رضہ کا عقد نکاح حضور صہ سے طے کرایا؟ / A: ان میں سے کوئی نہیں / B: حضرت جویریہ رضہ / C: حضرت صفیہ رضہ / D: حضرت خولہ بنت حکیم رضہ | Question: Through whom did Hazrat Abu Bakr Siddiq (RA) arrange the marriage contract of Hazrat Aisha (RA) with the Prophet (PBUH)? / A: None of these / B: Hazrat Juwayriya (RA) / C: Hazrat Safiyya (RA) / D: Hazrat Khawla bint Hakim (RA) | D | A | D |
| um_22216 | islamic studies | Question: الئن میں ____ ہے۔ / A: مد لازم کلمی مشقل / B: مد لازم حرفی مشقل / C: مد لازم کلمی مخفف / D: ان میں سے کوئی نہیں | Question: In "الئن" there is ____. / A: Madd Lazim Kalimi Muthaqqal / B: Madd Lazim Harfi Muthaqqal / C: Madd Lazim Kalimi Mukhaffaf / D: None of these | C | B | C |
| um_22310 | islamic studies | Question: ایتاء کے معنی کیا ہیں؟ / A: ان میں سے کوئی نہیں / B: اداکیا / C: آگیا / D: دینا | Question: What is the meaning of Ita? / A: None of these / B: Paid / C: Came / D: To give | D | A | D |
| um_22388 | islamic studies | Question: اسلام میں حقوق کی کتنی قسمیں ہیں؟ / A: دو / B: چھ / C: چار / D: تین | Question: How many types of rights are there in Islam? / A: Two / B: Six / C: Four / D: Three | A | C | A |
| um_22393 | islamic studies | Question: سورت الہمزہ میں ہے کہ اس شخص کے لیے خرابی اور ہلاکت ہے؟ / A: ان میں سے کوئی نہیں / B: جو نماز قضا کرتا ہے / C: جو مال جمع کرتا ہے / D: جو غصہ کرتا ہے | Question: In Surah Al-Humazah, for which person is there ruin and destruction? / A: None of these / B: Who misses the prayer / C: Who accumulates wealth / D: Who gets angry | C | A | C |
| um_22435 | islamic studies | Question: کتاب العصر کو کس نے مرتب کیا ہے؟ / A: Imam Shafi (رضہ) / B: Imam Abu Hanifah (رضہ) / C: Imam Malik (رضہ) / D: Imam Humbal (رضہ) | Question: Who compiled Kitab al-Asr? / A: Imam Shafi (RA) / B: Imam Abu Hanifah (RA) / C: Imam Malik (RA) / D: Imam Humbal (RA) | B | A | B |

### OTHER — S_ro right, S_ur wrong (200 candidates, 15 shown)

| row | subdomain | Urdu item | R1 translation | gold | S_ro | S_ur |
|---|---|---|---|---|---|---|
| um_00620 | chemistry | Question: جب امونیکل برائن سے CO2 کو گزارا جاتا ہے تو کونسے سالٹ کا رسوب بنتا ہے؟ / A: Na2CO3 / B: NaHCO3 / C: NH4HCO3 / D: (NH4)2CO3 | Question: When CO2 is passed through ammonical brine, which salt precipitate is formed? / A: Na2CO3 / B: NaHCO3 / C: NH4HCO3 / D: (NH4)2CO3 | B | B | C |
| um_01896 | general science | Question: جس ذہنی عارضے میں فرد احساسِ ذمہ داری سے محروم ہو جاتا ہے اسے کیا کہتے ہیں؟ / A: نیوراسس / B: سائیکوسس / C: اپیپٹوسس / D: ایسکری ایسس | Question: What is the mental disorder called in which an individual loses the sense of responsibility? / A: Neurosis / B: Psychosis / C: Epilepsy / D: Ascariasis | B | B | C |
| um_09793 | health and physical education | Question: تعاقب کرنے کا کھیل ہے: / A: فٹ بال / B: ہاکی / C: رومال گرانا / D: کرکٹ | Question: The game of chasing is: / A: Football / B: Hockey / C: Drop the Handkerchief / D: Cricket | C | C | A |
| um_10615 | civics | Question: بنیادی اصولوں کی کمیٹی کے ارکان کی تعداد تھی: / A: 10 / B: 20 / C: 14 / D: 24 | Question: The number of members of the Basic Principles Committee was: / A: 10 / B: 20 / C: 14 / D: 24 | D | D | C |
| um_11268 | geography | Question: مدو جزر اکبر قمری مہینے کی کن تواریخ کو پیدا ہوتے ہیں؟ / A: ساتویں اور چودھویں / B: پہلی اور چودھویں / C: چودھویں اور اکیسویں / D: پہلی اور ساتویں | Question: On which dates of the lunar month do spring tides occur? / A: Seventh and fourteenth / B: First and fourteenth / C: Fourteenth and twenty-first / D: First and seventh | B | B | D |
| um_11451 | geography | Question: نہر سویز ملاتی ہے؟ / A: بحری اوقیانوس اور بحرہ الکاہل / B: بحیرہ احمر اور بحیرہ روم / C: بحیرہ احمر اور بحر اوقیانوس / D: بحیرہ روم اور بحر الکاہل | Question: The Suez Canal connects? / A: Atlantic Ocean and Pacific Ocean / B: Red Sea and Mediterranean Sea / C: Red Sea and Atlantic Ocean / D: Mediterranean Sea and Pacific Ocean | B | B | C |
| um_12072 | current and international affairs | Question: پاکستان شریعت کورٹ اسلام آباد کے موجودہ چیف جسٹس کا کیا نام ہے؟ / A: جسٹس اقبال حمید الرحمن / B: جسٹس ڈاکٹر سید محمد انور / C: جسٹس شیخ انجم الحسن / D: جسٹس ظہور احمد شہوانی | Question: What is the name of the current Chief Justice of the Pakistan Shariat Court Islamabad? / A: Justice Iqbal Hameed ur Rahman / B: Justice Dr. Syed Muhammad Anwar / C: Justice Sheikh Anjum Hasan / D: Justice Zahoor Ahmed Shahwani | A | A | D |
| um_12948 | psychometrics | Question: مجھے دوسروں سے زیادہ پوچھ گچھ پسند نہیں ہے۔ / A: غیر متفق / B: سخت متفق / C: متفق / D: سخت غیر متفق | Question: I do not like being questioned more than others. / A: Disagree / B: Strongly agree / C: Agree / D: Strongly disagree | A | A | C |
| um_13029 | psychometrics | Question: میں تضادات کو تلاش کرنے پر گہری نظر رکھتا ہوں۔ / A: اختلاف کرنا / B: سختی سے متفق / C: متفق / D: غیر یقینی | Question: I keep a close eye on finding contradictions. / A: Disagree / B: Strongly agree / C: Agree / D: Uncertain | B | B | A |
| um_23843 | ethics | Question: کاروبار میں ____ کو پورا رکھیں: / A: ناپ تول / B: اخلاق / C: نفع / D: آداب | Question: Keep ____ complete in business: / A: Weights and measures / B: Morals / C: Profit / D: Etiquette | A | A | B |
| um_24007 | fine arts | Question: مندرجہ ذیل میں سے کون سا فن کا اصول ہے جو کسی بھی وقت ہوتا ہے جب کسی ٹکڑے کے عنصر کو فنکاروں کے ذریعہ غلبہ حاصل ہوتا ہے؟ / A: ان میں سے کوئی بھی نہیں۔ / B: زور / C: ہم آہنگی / D: رنگ | Question: Which of the following is a principle of art that occurs whenever an element of a piece is given dominance by the artist? / A: None of these. / B: Emphasis / C: Harmony / D: Color | B | B | A |
| um_24076 | fine arts | Question: اس بناوٹ کو جو رابطے کے احساس کے ساتھ محسوس کیا جاتا ہے اسے ____ کہا جاتا ہے؟ / A: سخت ساخت / B: ان میں سے کوئی بھی نہیں۔ / C: اصلی ساخت / D: کھردری ساخت | Question: The texture that is felt with the sense of touch is called ____? / A: Hard texture / B: None of these. / C: Actual texture / D: Rough texture | C | C | B |
| um_24890 | professional studies | Question: کپڑا تانے کو مختلف انداز سے ۔۔۔ کے تاروں سے گزار کر بنتا ہے : / A: بانے / B: روئیں / C: سوراخ / D: چمک | Question: Cloth is made by passing the warp in different ways through the threads of ... : / A: weft / B: cotton / C: hole / D: shine | A | A | C |
| um_24970 | professional studies | Question: شیر خوار بچوں کیلئے کون سے رنگ مناسب ہیں : / A: شوخ / B: ہلکے / C: صوفیانہ / D: کالے | Question: Which colors are suitable for infants: / A: Bright / B: Light / C: Mystical / D: Black | B | B | C |
| um_25236 | general knowledge | Question: حضرت محمد صلی اللہ علیہ وسلم نے کن صحابی رضی اللہ عنہ کے بارے میں کہا تھا کہ وہ بنی عامر میں سب سے عمدہ سواروں میں سے ہیں ؟ / A: حضرت ابو دجانہ رضی اللہ عنہ / B: حضرت ابو درداء رضی اللہ عنہ / C: حضرت سلیط بن عمرو رضی اللہ عنہ / D: حضرت ذوالجوش صبابی  ... | Question: About which companion (may Allah be pleased with him) did Prophet Muhammad (peace be upon him) say that he is among the best horsemen of Banu Amir? / A: Hazrat Abu Dujana (may Allah be pleased with him) / B: Hazrat Abu Darda (may Allah be pleased wit ... | D | D | A |

### OTHER — S_ro wrong, S_ur right (83 candidates, 15 shown)

| row | subdomain | Urdu item | R1 translation | gold | S_ro | S_ur |
|---|---|---|---|---|---|---|
| um_03091 | mathematics | Question: اگر A اور B دو متداخل سیٹ ہوں تو A ∩ B = ؟ / A: B ∪ A / B: A ∩ B / C: B ∩ A / D: φ | Question: If A and B are two overlapping sets, then A ∩ B = ? / A: B ∪ A / B: A ∩ B / C: B ∩ A / D: φ | C | B | C |
| um_04768 | physics | Question: فالکن (عقاب) کی سپیڈ ہوتی ہے: / A: 200 kmh−1 / B: 70 kmh−1 / C: 100 kmh−1 / D: 50 kmh−1 | Question: The speed of a falcon (eagle) is: / A: 200 kmh−1 / B: 70 kmh−1 / C: 100 kmh−1 / D: 50 kmh−1 | A | B | A |
| um_11222 | geography | Question: الپائن پہاڑوں کی عمر ____ ملین سال ہوتی ہے۔ / A: 280 / B: 25 / C: 500 / D: 400 | Question: The age of the Alpine mountains is ____ million years. / A: 280 / B: 25 / C: 500 / D: 400 | B | C | B |
| um_12106 | current and international affairs | Question: پاکستا ن کے موجود ہ وزیر دفاع کا نام بتائیں ؟ / A: شاہ محمود قریشی / B: شیخ رشید احمد / C: خواجہ محمد آصف / D: پرویز خٹک | Question: Name the current Defense Minister of Pakistan? / A: Shah Mahmood Qureshi / B: Sheikh Rasheed Ahmed / C: Khawaja Muhammad Asif / D: Pervez Khattak | C | A | C |
| um_12275 | current and international affairs | Question: سال 2025 کا پی ایس ایل فینل کس ٹیم نے جیتا؟ / A: لاہور قلندرز / B: اسلام آباد متحد / C: کوئٹہ گلیڈی ایٹرز / D: کراچی کے بادشاہ | Question: Which team won the PSL final of 2025? / A: Lahore Qalandars / B: Islamabad United / C: Quetta Gladiators / D: Karachi Kings | A | C | A |
| um_12583 | commerce | Question: مالک کے ذاتی استعمال کے لیے بینک سے نکلوانا جانے والا کیش درج ہوتا ہے، کیش بک میں: / A: ڈیبٹ سائیڈ / B: دونوں ڈیبٹ اور کریڈٹ سائیڈ / C: کسی سائیڈ / D: کریڈٹ سائیڈ | Question: Cash withdrawn from the bank for the owner's personal use is recorded in the cash book on: / A: Debit side / B: Both debit and credit side / C: Any side / D: Credit side | D | A | D |
| um_12704 | commerce | Question: چلت کھاتہ موزوں ہوتا ہے۔ / A: تنخواہ دار کے لیے / B: تمام صحیح ہیں / C: سرمایہ دار کے لیے / D: کاروباری کے لیے | Question: Current account is suitable. / A: For a salaried person / B: All are correct / C: For a capitalist / D: For a businessman | D | A | D |
| um_12729 | commerce | Question: رہن کی صورت میں جائیداد پر قبضہ ہوتا ہے : / A: مقروض کا / B: حکومت کا / C: قرض خواہ کا / D: یہ تمام | Question: In case of mortgage, the possession of the property is with: / A: The debtor's / B: The government's / C: The creditor's / D: All of these | C | A | C |
| um_12956 | psychometrics | Question: میں ثقافت پسند ہوں ۔ / A: سخت متفق / B: غیر متفق / C: متفق / D: سخت غیر متفق | Question: I am a traditionalist. / A: Strongly agree / B: Disagree / C: Agree / D: Strongly disagree | C | A | C |
| um_12987 | psychometrics | Question: سر میں اکثر درد رہتا ہے۔ / A: بالکل غلط / B: بالکل صحیح / C: صحیح / D: غلط | Question: I often have a headache. / A: Completely wrong / B: Completely correct / C: Correct / D: Wrong | A | C | A |
| um_13018 | psychometrics | Question: میں ٹیم ورک کے ساتھ ٹیم کے اہداف اور مقاصد حاصل کرنے کا رجحان رکھتا ہوں۔ / A: متفق / B: اختلاف کرنا / C: سختی سے متفق / D: غیر یقینی | Question: I tend to achieve team goals and objectives through teamwork. / A: Agree / B: Disagree / C: Strongly agree / D: Uncertain | C | A | C |
| um_13079 | psychometrics | Question: ایک ٹریلین کتنے ملین کے برابر ہے؟ / A: ان میں سے کوئی نہیں / B: لاکھوں / C: اربوں کروڑوں / D: لاکھوں کروڑوں | Question: How many millions is one trillion equal to? / A: None of these / B: Millions / C: Billions and crores / D: Millions and crores | D | A | D |
| um_23854 | ethics | Question: ایک ملازم کے لیے سب سے اہم ہے۔ / A: یہ تمام / B: عزت نفس / C: مشاہرہ / D: عہدہ | Question: For an employee, the most important thing is. / A: All of these / B: Self-respect / C: Salary / D: Position | B | A | B |
| um_23990 | fine arts | Question: سپرش کا کیا مطلب ہے؟ / A: آسانی سے دیکھا جاتا ہے۔ / B: چھونے کا حساس / C: چھونا ناممکن / D: ان میں سے کوئی بھی نہیں۔ | Question: What does sparse mean? / A: Easily seen. / B: Sensitive to touch / C: Impossible to touch / D: None of these. | B | D | B |
| um_24555 | professional development | Question: کراچی ڈویژن میں کل کتنے اضلاع شامل ہیں؟ / A: 06 / B: 05 / C: 07 / D: 04 | Question: How many districts are included in total in Karachi Division? / A: 06 / B: 05 / C: 07 / D: 04 | C | B | C |

