# D11 DEV200 supplement (descriptive)

Written by `eval/error_analysis_tests/efbpt/d11_dev200_supplement.py` (CPU, no model, no generation re-run) from sealed DEV200 files only; nothing under any eval458 directory was opened. Descriptive: no gate, no decision and no eval458 rule depends on anything here. Conventions as in `docs/handoff/D11_FULL_READOUT.md`: b = the first-named arm right and the second wrong, c = the reverse, p = exact two-sided McNemar (`d5_balanced_extract.mcnemar`), unparsed = wrong, n = 200 (87 gold yes / 113 gold no) unless a subset is named.

Inputs (MD5 asserted): A_r2 `7ccec942`; TF_v1 `c54b7553`; TR `bab987cc`; TR2 `17f1c43d`; TR_L1 `b8db599f`; TR_wrong `9519b9ad`; GF `448fec41`; TO1 `6548b408`; readings R1 `1f5b2db3`, R2 `cd77944e`; teacher facts R1 `160406e1`. The frozen extractor reproduces the sealed `pred` on all 1600 generations read.

## S1. Scorer sensitivity: frozen extractor vs first-mention variant

The variant is the frozen extractor with one change: in Rule 2, where the frozen rule takes the **last** ہاں/نہیں mention of the answer segment, the **first** is taken. Rule 1 (JSON), the cut after the last "حتمی جواب" marker, Rule 3 (English) and the plain substring matching are unchanged (so a word such as یہاں or وہاں, which contains ہاں, counts as a mention under both rules). The set of unparsed rows is therefore the same under both.

| arm | frozen acc % (correct) | first-mention acc % (correct) | variant − frozen pp | rows where the two labels differ | of those: frozen right / variant right | unparsed |
|---|---|---|---|---|---|---|
| A_r2 | 60.00 (120) | 55.00 (110) | -5.0 | 38 | 24 / 14 | 8 |
| TF_v1 | 65.00 (130) | 67.50 (135) | +2.5 | 33 | 14 / 19 | 1 |
| TR | 68.50 (137) | 72.00 (144) | +3.5 | 27 | 10 / 17 | 3 |
| TR2 | 65.00 (130) | 69.50 (139) | +4.5 | 33 | 12 / 21 | 3 |
| TR_L1 | 67.00 (134) | 70.50 (141) | +3.5 | 39 | 16 / 23 | 2 |
| TR_wrong | 50.50 (101) | 51.50 (103) | +1.0 | 22 | 10 / 12 | 11 |
| GF | 76.00 (152) | 77.00 (154) | +1.0 | 28 | 13 / 15 | 2 |
| TO1 | 74.00 (148) | 78.00 (156) | +4.0 | 36 | 14 / 22 | 2 |

| TR vs A_r2 | acc TR | acc A_r2 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|
| frozen extractor (the sealed result) | 68.50 | 60.00 | +8.5 | 39 | 22 | 0.03962 |
| first-mention variant | 72.00 | 55.00 | +17.0 | 57 | 23 | 0.0001831 |

Rows whose label is the same under both rules in both arms: 142 of 200. The frozen extractor is the pre-registered scorer; the variant is a sensitivity reading only.

## S2. TR vs A_r2 on parsed and untruncated rows

| rows | n | acc TR | acc A_r2 | Δ pp (on these rows) | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| all rows (the sealed result) | 200 | 68.50 | 60.00 | +8.50 | 39 | 22 | 0.03962 |
| parsed in both arms | 189 | 69.84 | 63.49 | +6.35 | 34 | 22 | 0.1409 |
| parsed and untruncated in both arms | 173 | 69.36 | 62.43 | +6.94 | 31 | 19 | 0.1189 |

Rows dropped: A_r2 unparsed 8, TR unparsed 3 (either 11); A_r2 truncated 11, TR truncated 9; unparsed or truncated in either arm 27.

| discordant rows | n | other arm unparsed | other arm truncated | unparsed or truncated | both |
|---|---|---|---|---|---|
| 39 wins (TR right, A_r2 wrong): A_r2 was | 39 | 5 | 3 | 6 | 2 |
| 22 losses (A_r2 right, TR wrong): TR was | 22 | 0 | 2 | 2 | 0 |

## S3. Paired against the D10 arm TF_v1

TF_v1 = the D10 teacher-facts arm (same teacher, same student, same layout B; teacher input = the Urdu question, no reading stage).

| comparison | acc first | acc TF_v1 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|
| TR vs TF_v1 | 68.50 | 65.00 | +3.5 | 26 | 19 | 0.3713 |
| TR_L1 vs TF_v1 | 67.00 | 65.00 | +2.0 | 29 | 25 | 0.6835 |

## S4. Reader agreement (R1 vs R2) as a deployable proxy

Agreement = the R1 (api_qwen3.8-max) and R2 (api_deepseek-v4-pro) readings are equal after `norm()` (`d11r_score.norm`: "_" → space, strip, lower-case, collapse whitespace). It needs no gold. Row-M1 = share of the gold question's proper nouns recovered by the R1 reading (`d11r_score.m1_row`; 45 rows have no gold proper noun). TR uses the R1 reading in both groups.

| R1 vs R2 reading | n | R1 row-M1 = 1 | row-M1 < 1 | no proper noun | share row-M1 = 1 (of n) | share row-M1 = 1 (of rows with a proper noun) | TR acc % | A_r2 acc % | Δ pp | wins | losses | exact two-sided p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agree after norm() | 74 | 54 | 6 | 14 | 73.0% | 90.0% | 67.57 | 52.70 | +14.86 | 16 | 5 | 0.0266 |
| disagree | 126 | 70 | 25 | 31 | 55.6% | 73.7% | 69.05 | 64.29 | +4.76 | 23 | 17 | 0.4296 |
| all | 200 | 124 | 31 | 45 | 62.0% | 80.0% | 68.50 | 60.00 | +8.50 | 39 | 22 | 0.03962 |

Post-hoc split of one run; the two groups differ in question difficulty as well as in reading quality (see the A_r2 column).

## S5. The first 15 of 39 TR-vs-A_r2 WIN rows (TR right, A_r2 wrong)

Format of readout §8.1: qid, gold, Urdu question, gold question_en, the R1 reading (row-M1 in parentheses; "-" = no gold proper noun), the first two teacher facts, A_r2 answer, TR answer; answers show the frozen extractor's `pred` in bold and the first ~110 characters of the generation (whitespace collapsed). Rows in DEV200 file order.

| # | qid | gold | Urdu question | gold question_en | R1 reading (row-M1) | fact 1 | fact 2 | A_r2 | TR |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `05e9cdc44f1b066badd7` | no | کیا خواتین میں مونچھیں ہونا عام بات ہے؟ | Is it common for women to have moustaches? | Is it normal for women to have a mustache? (-) | Women are adult human females. | A mustache is a growth of hair on the upper lip. | **yes** — ہاں، خواتین میں مونچھیں ہونا عام ہے۔ یہ ایک طبی طبیعت ہے جو کچھ خواتین میں پیدا ہوتی ہے۔ مونچھیں کی ترقی ہارمو | **no** — نہیں، خواتین میں مونچھیں ہونا عام بات نہیں ہے۔ عام طور پر، خواتین کے چہرے پر ہیلیوس ہیئر (vellus hair) ہوتا ہے |
| 2 | `064d8faaa8f164b55270` | yes | کیا اپالو 15 کے عملے کو ایک پہیے والی سائیکل چلانے میں دشواری ہوگی؟ | Would the crew of Apollo 15 have difficulty riding a unicycle? | Would the crew of Apollo 15 have difficulty riding a unicycle? (1.00) | Apollo 15 was the fourth manned mission to land on the Moon. | The crew of Apollo 15 consisted of David Scott, James Irwin, and Alfred Worden. | **no** — اپولو 15 کے عملے کو ایک پہیے والی سائیکل چلانے میں دشواری ہو سکتی ہے، کیونکہ ایک پہیے والی سائیکل چلانا دو پہی | **yes** — ہاں، اپولو 15 کے عملے کو ایک پہیے والی سائیکل (Unicycling) چلانے میں دشواری ہوسکتی ہے۔ یہ دشواری ان عوامل کی ب |
| 3 | `0fe3edf35bd1a1e662a9` | no | کیا گینڈالف فرضی طور پر چارمد کے بارباس کے لیے ایک زبردست دشمن ہے؟ | Is Gandalf hypothetically a formidable foe for Charmed's Barbas? | Would Gandalf hypothetically be a formidable foe for Charmed's Barbas? (1.00) | Gandalf is a powerful wizard in J.R.R. Tolkien's Middle-earth legendarium. | Gandalf is known for his magical abilities, including fire magic and illusions. | **yes** — ہاں، گینڈالف (Gendarme) فرضی طور پر چارمد (Charmode) کے بارباس (Barbar) کے لیے ایک زبردست دشمن ہے۔ یہ ایک گیم  | **no** — ہاں، گینڈالف فرضی طور پر چارمد کے بارباس کے لیے ایک زبردست دشمن ہوسکتا ہے۔ اس کی وجہ یہ ہے کہ گینڈالف ایک بہت  |
| 4 | `27368c21e50b6af694ab` | no | کیا ایک اخلاقیات کے پروفیسر سیزان پر کلاس پڑھائیں گے؟ | Would an ethics professor teach a class on Cezanne? | Will an ethics professor teach a class on Suzanne? (0.00) | Ethics is a branch of philosophy that involves systematizing, defending, and recommending concepts of right and wrong conduct. | A professor is an academic rank at universities and other post-secondary education and research institutions in most countries. | **yes** — ہاں، ایک اخلاقیات کے پروفیسر کو کلاس پڑھانے کا کام کر سکتا ہے۔ اس کی صلاحیت، تعلیم، تربیت اور تجربے کے برمحل ہ | **no** — اس سوال کا جواب یہ ہے: **نہیں، ایک اخلاقیات کے پروفیسر Suzanne نہیں پڑھائیں گے۔** اس کی وضاحت یہ ہے کہ: - "Suz |
| 5 | `3486b6b5c69878bb088d` | no | کیا ایریزونا میں کیلیفورنیا کے مقابلے میں زیادہ گوبھی اگتی ہے؟ | Do more Cauliflower grow in Arizona than California? | Does Arizona grow more cabbage than California? (0.67) | Arizona is a state in the southwestern United States. | California is a state on the West Coast of the United States. | **yes** — اس سوال کا جواب یہ ہے کہ **کیلیفورنیا** میں **گوبھی** (Cabbage) کی فصل کی پیداوار **ایریزونا** کے مقابلے میں * | **no** — اس سوال کا جواب یہ ہے: **نہیں، ایریزونا میں کیلیفورنیا کے مقابلے میں زیادہ گوبھی اگتی نہیں**۔ **دلیل:** - کیلی |
| 6 | `3503e8d41364eb0b33bc` | no | کیا ایک ذمہ دار بارٹینڈر میلی بوبی براؤن کے لیے مشروب بنائے گا؟ | Would a responsible bartender make a drink for Millie Bobby Brown? | Would a responsible bartender make a drink for Millie Bobby Brown? (1.00) | Millie Bobby Brown is a British actress and model. | Millie Bobby Brown was born on February 19, 2004. | **None (unparsed)** — اس سوال کا جواب یہ ہے کہ **ایک ذمہ دار بارٹینڈر میلی بوبی براؤن کے لیے مشروب بنائے گا**۔ کیونکہ میلی بوبی براؤ | **no** — نہیں، ایک ذمہ دار بارٹینڈر میلی بوبی براؤن کے لیے مشروب نہیں بنائے گا۔ اس کی وجہ یہ ہے کہ میلی بوبی براؤن 2004 |
| 7 | `3b38e00d8d1da26f7f02` | no | کیا ماریو تیز دوڑنے کے لیے مشروم استعمال کرتا ہے؟ | Does Mario use mushrooms to run faster? | Does Mario use a mushroom to run fast? (1.00) | Mario is a fictional character in the Super Mario video game series created by Nintendo. | In the Super Mario series, Mario can use power-ups to gain new abilities. | **yes** — ہاں، ماریو تیز دوڑنے کے لیے مشروم استعمال کرتا ہے۔ یہ ایک مشہور ویڈیو گیم "سوپر ماریو" میں دیکھا جاتا ہے جہاں  | **no** — نہیں، ماریو تیز دوڑنے کے لیے مشروم استعمال نہیں کرتا۔ Super Mushroom کا استعمال ماریو کو بڑا ہو جانے اور ایک ا |
| 8 | `3d7050c0e9bfbeef27c1` | yes | کیا میلانیا ٹرمپ کا پیشہ اولگا کیوریلنکو کے پیشے کے برابر تھا؟ | Did Melania Trump have same profession as Olga Kurylenko? | Was Melania Trump's profession the same as Olga Kurylenko's profession? (1.00) | Melania Trump is a former model and the wife of Donald Trump, the 45th President of the United States. | Melania Trump began her modeling career at a young age and worked with several prominent modeling agencies. | **no** — اس سوال کا جواب یہ ہے کہ **میلانیا ٹرمپ** اور **اولگا کیوریلنکو** کے پیشے مختلف ہیں، اور ان کے کام کے حوالے سے | **yes** — جی ہاں، میلانیا ٹرمپ اور اولگا کیوریلنکو دونوں نے ماڈلنگ کے شعبے میں کام کیا ہے، اور ان کا پیشہ ماڈلنگ تھا۔ اس |
| 9 | `3f5af97fdff20028347b` | no | کیا فرانسیسی انقلاب ٹیلیویژن پر دکھایا گیا تھا؟ | Was the French Revolution televised? | Was the French Revolution shown on television? (1.00) | The French Revolution was a period of radical social and political upheaval in France that lasted from 1789 to 1799. | Television is a telecommunication medium used for transmitting moving images in monochrome or in color, and in two or three dimensions. | **yes** — ہاں، فرانسیسی انقلاب کے مختلف پہلوؤں کو ٹیلیویژن پر دکھایا گیا ہے۔ ٹیلیویژن کے ذریعے تاریخی واقعات کو دکھانے ک | **no** — نہیں، فرانسیسی انقلاب ٹیلیویژن پر دکھایا گیا تھا۔ اس کی وجہ یہ ہے کہ فرانسیسی انقلاب 1789 سے 1799 تک ہوا، جبکہ |
| 10 | `4760fe3d0c80beb30edd` | no | اگر کوئی شخص ویگن ہو تو کیا وہ شہد کھائے گا؟ | If someone is a vegan, would they eat honey? | If a person is vegan, will they eat honey? (-) | Veganism is a lifestyle and dietary choice that seeks to exclude, as far as possible, all forms of exploitation of animals for food, clothing, and other purposes. | Vegans do not consume or use products that come from animals, including eggs, dairy products, and even gelatin. | **yes** — ہاں، ویگن ہونے کا مطلب یہ نہیں کہ کوئی شخص شہد کھانے سے گریز کرے۔ ویگن ڈائیٹ میں جانوروں کی پیداوار سے حاصل ہو | **no** — اس سوال کا جواب یہ ہے کہ **ویگن شخص کے لیے شہد کھانے کا فیصلہ ان کی ذاتی ایمانداری اور ویگنزم کی تعریف پر منحص |
| 11 | `4ec8cd118be8a53ec516` | no | کیا الیگزینڈر ہیملٹن کو کوالاز کے بارے میں معلوم ہوتا؟ | Would Alexander Hamilton have known about koalas? | Would Alexander Hamilton have known about Koalas? (1.00) | Alexander Hamilton was a founding father of the United States and lived from 1755 to 1804. | Koalas are native to Australia. | **yes** — ہاں، الیگزینڈر ہیملٹن کو کوالاز کے بارے میں معلوم ہوتا تھا۔ ہیملٹن ایک امریکی سیاست دان، قانونی ماہر، اور امری | **no** — نہیں، الیگزینڈر ہیملٹن کو کوالاز کے بارے میں معلوم نہیں ہوتا۔ اس کی وجہ یہ ہے کہ ہیملٹن کی زندگی (1755–1804) ک |
| 12 | `502d4dae6f08e73a5569` | yes | کیا ہوم ڈیپو سلیب پر چڑھانے کے سامان کے لیے ایک جامع دکان ہے؟ | Is Home Depot a one stop shop for crucifixion supplies? | Is Home Depot a one-stop shop for slab lifting equipment? (1.00) | The Home Depot is an American multinational home improvement retail corporation. | The Home Depot operates a chain of big-box stores offering a wide range of products and services for home improvement. | **no** — ہاں، ہوم ڈیپو (Home Depot) چڑھانے کے سامان کے لیے ایک جامع دکان ہے۔ یہ امریکا کی سب سے بڑی ہوم ڈیپو چینل ہے جو | **yes** — ہاں، ہوم ڈیپو سلیب لیولنگ یا سلیب پر چڑھانے کے سامان کے لیے ایک جامع دکان ہے۔ ہوم ڈیپو کے پاس سیمنٹ، ماسٹریکس، |
| 13 | `5e2d9519c7c521411817` | yes | کیا ٹی ایم این ٹی رنگ بھرنے کی کتاب میں پیزا ہوگا؟ | Would a TMNT coloring book have pizza in it? | Will there be pizza in the TMNT coloring book? (1.00) | Teenage Mutant Ninja Turtles is a popular American media franchise created by Kevin Eastman and Peter Laird. | The Teenage Mutant Ninja Turtles are known for their love of pizza. | **None (unparsed)** — اس سوال کا جواب یہ ہے کہ **ٹی ایم این ٹی (TNT)** ایک **انفیکٹیوز گیس** ہے جو **بلاک بوسٹر** یا **انفیکٹیوز ماد | **yes** — ہاں، ٹی ایم این ٹی (Teenage Mutant Ninja Turtles) کی رنگ بھرنے والی کتاب میں پیزا ہو سکتا ہے۔ یہ اس لیے ممکن ہ |
| 14 | `67079beff747978ed4c2` | no | کیا ایم ایف ڈوم فینٹاسٹک فور کا ولن ہے؟ | Is MF Doom a Fantastic Four villain? | Is MF Doom a villain of the Fantastic Four? (1.00) | MF DOOM was a British-American rapper and record producer. | MF DOOM was known for his comic book-inspired lyrics and aliases. | **yes** — ہاں، **ایم ایف ڈوم** (M.F. Doom) یا **دومن** (Doom) **فینٹاسٹک فور** (Fantastic Four) کا ایک اہم **ولن** ہے۔ * | **no** — نہیں، ایم ایف ڈوم (MF DOOM) فینٹاسٹک فور کا ولن نہیں ہے۔ MF DOOM ایک ریپر اور ریکارڈ پروڈیوسر تھا جو اپنی کامی |
| 15 | `690a36d5fed5b17969aa` | no | کیا جیک ڈیمپسی نے موجودہ WBC ہیوی ویٹ چیمپئن سے مقابلہ کیا تھا؟ | Did Jack Dempsey fight the current WBC heavyweight champion? | Did Jack Dempsey fight the current WBC heavyweight champion? (1.00) | Jack Dempsey was an American professional boxer who competed from 1914 to 1927. | Jack Dempsey held the World Heavyweight Championship from 1919 to 1926. | **yes** — جی ہاں، جیک ڈیمپسی (Jake Dempsey) نے موجودہ WBC ہیوی ویٹ چیمپئن سے مقابلہ کیا ہے۔ اگر آپ کے سوال میں "موجودہ"  | **no** — نہیں، جیک ڈیمپسی نے موجودہ WBC ہیوی ویٹ چیمپئن سے مقابلہ نہیں کیا۔ اس کی دو بنیادی وجوہات ہیں: 1. **زمانی فرق* |

Cross-check: 15 of these 15 rows are character-identical to the rows of readout §8.2 (`docs/handoff/D11_FULL_READOUT.md`).
