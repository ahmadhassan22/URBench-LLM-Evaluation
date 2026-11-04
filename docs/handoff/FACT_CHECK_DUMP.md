# Fact-check dump: the three V5 hand-check samples (90 facts)

Read-only. AUDIT FILE: nothing in this file enters any prompt. It joins each sampled fact to the DEV200 gold row (`data/strategyqa_official/dev200_seed4242.jsonl`: `question_en`, `answer`, `urbench_facts`) so a human can mark each fact true/false and relevant/irrelevant. The three samples were drawn by the D8, D10 and D10v2 scripts with `random.Random(8).sample(..., 30)` over the facts injected in the Xp / TF arm; the D10 files carry `truth: null`, `relevance: null` to be filled by hand, the D8 sample has `status: PENDING hand check` in `d8_summary.json`. A qid may appear twice in one sample (two of its facts drawn). Gold `answer` is shown as yes/no from the boolean.

| sample | source | n | distinct qids |
|---|---|---|---|
| D8-Xp | `outputs/efbpt/d8/d8_summary.json` | 30 | 28 |
| D10-TF | `outputs/efbpt/d10/student/tf_facts_sample30.jsonl` | 30 | 29 |
| D10v2-TF | `outputs/efbpt/d10v2/student/tf_facts_sample30.jsonl` | 30 | 29 |

Scoring columns to fill by hand: **truth** (T = true, F = false, U = unverifiable) and **relevance** (R = the fact bears on the question's bridge, N = not). Record verdicts in a separate file; this dump is generated and should not be edited.

## Sample D8-Xp

Source: `outputs/efbpt/d8/d8_summary.json → V5_Xp_hand_check_sample.facts (seed 8, n 30, job 108196; there is no separate d8_xp_sample30.jsonl file)`

### D8-Xp #1 — qid `4e1b65e81ec09397b26e`

- **Fact to check:** The giant squid has eight arms and two longer tentacles.
- Context from the sample row: titles_corpus: Giant squid
- Gold question_en (audit): Is capturing giant squid in natural habitat impossible with no gear?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Giant squids live between 1,000 and 3,800 feet in the ocean.
  - With a dry dive suit, a scuba tank, gloves, and so on, divers can reach depths of around 1000 feet.
  - Without scuba gear people can safely dive a maximum of 60 feet without feeling the negative risks associated with diving beyond the limit.
- truth: ____   relevance: ____

### D8-Xp #2 — qid `80aa769f55b14c1e4d8d`

- **Fact to check:** Sable is primarily found in forest environments of Russia, Siberia, and northern Mongolia.
- Context from the sample row: titles_corpus: Sable
- Gold question_en (audit): Are sables related to wolverines?
- Gold answer: yes
- Gold urbench_facts (audit):
  - The sable is a species of marten, which make up the genus Martes.
  - Wolverines are from the genus Gulo.
  - Both the Martes and the Gulo are from the family Mustelidae.
- truth: ____   relevance: ____

### D8-Xp #3 — qid `86ead3f15417204affc2`

- **Fact to check:** Dustin Hoffman was born in Los Angeles, California.
- Context from the sample row: titles_corpus: Dustin Hoffman
- Gold question_en (audit): Is Dustin Hoffman one of the B'nei Yisrael?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Dustin Hoffman was raised in a Jewish family.
  - In modern Hebrew, b'nei yisrael ("children of Israel") can denote the Jewish people at any time in history.
- truth: ____   relevance: ____

### D8-Xp #4 — qid `2bc94c6dfc79d7a86fa2`

- **Fact to check:** He has guest-hosted Saturday Night Live seven times.
- Context from the sample row: titles_corpus: Christopher Walken
- Gold question_en (audit): Could Christopher Walken enlist in the United States Marine Corps?
- Gold answer: no
- Gold urbench_facts (audit):
  - Christopher Walken is 77 years old.
  - The maximum age for first-time enlistment in the United States Marine Corps is 28.
- truth: ____   relevance: ____

### D8-Xp #5 — qid `42c947475244ed02629e`

- **Fact to check:** Pearl farmers can culture a pearl by placing a nucleus inside the oyster.
- Context from the sample row: titles_corpus: Oyster
- Gold question_en (audit): Can oysters be used in guitar manufacturing?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Oysters produce nacre
  - Nacre is also known as mother of pearl
  - Mother of pearl is commonly used as an inlay on guitar fretboards, headstocks, and soundboards
- truth: ____   relevance: ____

### D8-Xp #6 — qid `f7a32d14efd589015fe1`

- **Fact to check:** HSV-1 and HSV-2 are very common and contagious.
- Context from the sample row: titles_corpus: Herpes simplex virus; Virus
- Gold question_en (audit): Can Herpes simplex virus spread on Venus?
- Gold answer: no
- Gold urbench_facts (audit):
  - Herpes simplex virus is a disease that has the structure of a tiny protein cage.
  - Venus is the hottest planet and its temperature can reach 900°F.
  - Proteins lose their structure and break down at temperatures above 105.8°F.
- truth: ____   relevance: ____

### D8-Xp #7 — qid `068379affae04debca2b`

- **Fact to check:** The war began on 13 February 1996 with an insurgency by the CPN (Maoist).
- Context from the sample row: titles_corpus: Nepalese Civil War
- Gold question_en (audit): Did the Nepalese Civil War take place near India?
- Gold answer: yes
- Gold urbench_facts (audit):
  - The Nepalese Civil War happened in Nepal.
  - Nepal is a country that shares a border with India.
- truth: ____   relevance: ____

### D8-Xp #8 — qid `15885efe6f91724c16d9`

- **Fact to check:** The sound barrier refers to the large increase in aerodynamic drag when an object approaches the speed of sound.
- Context from the sample row: titles_corpus: Audi R8; Sound barrier
- Gold question_en (audit): Is sound barrier too much for Audi R8 V-10 Plus to break?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Audi R8 V-10 Plus is the fastest car produced by Audi.
  - The Audi R8 V-10 Plus has a top speed of 205 MPH.
  - To break the sound barrier, a jet must reach a speed of 770 MPH.
- truth: ____   relevance: ____

### D8-Xp #9 — qid `3295844627a9bd1b9135`

- **Fact to check:** The total value of the Harry Potter franchise was estimated at $25 billion in 2016.
- Context from the sample row: titles_corpus: Harry Potter
- Gold question_en (audit): Was Harry Potter a better investment than The Matrix for Warner Bros.?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Warner Bros. distributes several movie franchises including The Matrix, Harry Potter, and The Dark Knight.
  - The Matrix had 2 sequels.
  - Harry Potter had 7 sequels and several spin-offs.
  - Harry Potter and the Deathly Hallows – Part 2 is Warner Bros. highest grossing film worldwide with a box office gross of $1,342,932,398.
- truth: ____   relevance: ____

### D8-Xp #10 — qid `5ab66b27a0caee886970`

- **Fact to check:** Miami has a population of 442,241 as of the 2020 census.
- Context from the sample row: titles_corpus: Miami
- Gold question_en (audit): Would it be common to find a penguin in Miami?
- Gold answer: no
- Gold urbench_facts (audit):
  - Penguins are native to the deep, very cold parts of the southern hemisphere.
  - Miami is located in the northern hemisphere and has a very warm climate.
- truth: ____   relevance: ____

### D8-Xp #11 — qid `ae0891799990442fbb0e`

- **Fact to check:** Land use planning involves regulating land use, development, and subdivision to achieve urban planning goals.
- Context from the sample row: titles_corpus: Urban planner
- Gold question_en (audit): Would Paul Bunyan hypothetically be a poor choice for an urban planner?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Paul Bunyan was a legendary giant from tall tale stories.
  - Urban planners need to design buildings and ceilings in cities.
  - Paul Bunyan lived in the country side with a giant blue ox.
  - New York city apartment ceilings average around 8 feet in height.
  - Paul Bunyan was over seven feet tall.
- truth: ____   relevance: ____

### D8-Xp #12 — qid `48da75d87c66754ccc2e`

- **Fact to check:** The attack aimed to prevent the U.S. Pacific Fleet from interfering.
- Context from the sample row: titles_corpus: Attack on Pearl Harbor
- Gold question_en (audit): Was only woman to serve as U.S. Speaker of the House alive during the attack on Pearl Harbor?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Nancy Pelosi is the only woman to ever serve as Speaker of the United States House of Representatives.
  - Nancy Pelosi was born on Mar 26, 1940
  - The attach on Pearl Harbor occurred on December 7, 1941
- truth: ____   relevance: ____

### D8-Xp #13 — qid `8d708927b1593d8b9b42`

- **Fact to check:** The decision upheld the "separate but equal" doctrine.
- Context from the sample row: titles_corpus: Plessy v. Ferguson
- Gold question_en (audit): Would Woodrow Wilson support Plessy v. Ferguson decision?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Plessy v Ferguson was a landmark case that stated segregation did not violate the constitution.
  - President Woodrow Wilson escalated the discriminatory hiring policies and segregation of government offices.
  - By the end of 1913, under President Wilson, many departments, including the navy, had segregated work spaces, restrooms, and cafeterias.
- truth: ____   relevance: ____

### D8-Xp #14 — qid `de62d38623fde2c2285a`

- **Fact to check:** Viewership dropped significantly in the show's third season.
- Context from the sample row: titles_corpus: Ugly Betty
- Gold question_en (audit): Would a week be enough time to watch every episode of Ugly Betty?
- Gold answer: yes
- Gold urbench_facts (audit):
  - There are 168 hours in a week.
  - The entire Ugly Betty series is 85 hours long.
- truth: ____   relevance: ____

### D8-Xp #15 — qid `064d8faaa8f164b55270`

- **Fact to check:** Apollo 15 used the Lunar Roving Vehicle for lunar surface exploration.
- Context from the sample row: titles_corpus: Apollo 15; Unicycle
- Gold question_en (audit): Would the crew of Apollo 15 have difficulty riding a unicycle?
- Gold answer: yes
- Gold urbench_facts (audit):
  - There were 3 astronauts in the crew of the Apollo 15 mission.
  - A unicycle only contains one saddle, and is typically only operated by a single person.
- truth: ____   relevance: ____

### D8-Xp #16 — qid `9b491a31c4212e70b18c`

- **Fact to check:** Alfred Adler developed the theory of birth order and its impact on children's personality.
- Context from the sample row: titles_corpus: Middle child syndrome
- Gold question_en (audit): Did Jane Austen suffer from middle child syndrome?
- Gold answer: no
- Gold urbench_facts (audit):
  - Jane Austen was the second youngest of 8 children.
  - Middle child syndrome is the feeling of exclusion by middle children, due directly to their placement in their family's birth order.
- truth: ____   relevance: ____

### D8-Xp #17 — qid `a2dfceff60d5b18fc70d`

- **Fact to check:** It was developed by American breeder Ed Currie.
- Context from the sample row: titles_corpus: Carolina Reaper
- Gold question_en (audit): Would Carolina Reaper decrease sales if added to all US salsa?
- Gold answer: yes
- Gold urbench_facts (audit):
  - On average, Americans prefer milder salsa than Europeans.
  - The Carolina Reaper is the hottest pepper in the world. 
  - The Carolina Reaper is rated as 2,200,000 Scoville Heat Units.
- truth: ____   relevance: ____

### D8-Xp #18 — qid `9883d6c816d4855035ea`

- **Fact to check:** The bulb of the onion is at the base of the plant.
- Context from the sample row: titles_corpus: Onion; Tree
- Gold question_en (audit): Do onions have a form that resembles the inside of a tree?
- Gold answer: yes
- Gold urbench_facts (audit):
  - When bisected, an onion has rings that extend from the core to the outside.
  - Trees are formed of a series of rings that extend from the inside to the outside
- truth: ____   relevance: ____

### D8-Xp #19 — qid `8d06b619a7045ed02f51`

- **Fact to check:** She was made a Dame in 1999.
- Context from the sample row: titles_corpus: Shirley Bassey
- Gold question_en (audit): Is the title of Shirley Bassey's 1971 diamond song a true statement?
- Gold answer: no
- Gold urbench_facts (audit):
  - Shirley Bassey recorded the song Diamonds are Forever in 1971,
  - Over time, diamonds degrade and turn into graphite.
  - Graphite is the same chemical composition found in pencils.
- truth: ____   relevance: ____

### D8-Xp #20 — qid `a54b16376f82a6e14bf9`

- **Fact to check:** The current composition of Earth's atmosphere is the result of billions of years of biochemical modification by living organisms.
- Context from the sample row: titles_corpus: Atmosphere; Atmosphere of Earth
- Gold question_en (audit): Will a rock float in the atmosphere of Earth?
- Gold answer: no
- Gold urbench_facts (audit):
  - Things only float if they are less dense than the surrounding material.
  - Rocks are denser than air.
  - The atmosphere of Earth is made up of air.
- truth: ____   relevance: ____

### D8-Xp #21 — qid `c921bdade889def26cf4`

- **Fact to check:** Salt is primarily composed of sodium chloride (NaCl).
- Context from the sample row: titles_corpus: Isopropyl alcohol; Salt; Solubility
- Gold question_en (audit): In isopropyl alcohol, is the solubility of salt low?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Isopropyl alcohol is unique in the sense that salt remains visible.
  - When salt has high solubility, it becomes invisible.
- truth: ____   relevance: ____

### D8-Xp #22 — qid `42c947475244ed02629e`

- **Fact to check:** Windowpane oysters are harvested for their translucent shells.
- Context from the sample row: titles_corpus: Oyster
- Gold question_en (audit): Can oysters be used in guitar manufacturing?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Oysters produce nacre
  - Nacre is also known as mother of pearl
  - Mother of pearl is commonly used as an inlay on guitar fretboards, headstocks, and soundboards
- truth: ____   relevance: ____

### D8-Xp #23 — qid `8d708927b1593d8b9b42`

- **Fact to check:** The decision was later weakened by Brown v. Board of Education in 1954.
- Context from the sample row: titles_corpus: Plessy v. Ferguson
- Gold question_en (audit): Would Woodrow Wilson support Plessy v. Ferguson decision?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Plessy v Ferguson was a landmark case that stated segregation did not violate the constitution.
  - President Woodrow Wilson escalated the discriminatory hiring policies and segregation of government offices.
  - By the end of 1913, under President Wilson, many departments, including the navy, had segregated work spaces, restrooms, and cafeterias.
- truth: ____   relevance: ____

### D8-Xp #24 — qid `16c08e5f354b035c41a5`

- **Fact to check:** The Godfather stars Marlon Brando, Al Pacino, James Caan, and others.
- Context from the sample row: titles_corpus: The Godfather; USB flash drive
- Gold question_en (audit): Was a USB flash drive used in The Godfather?
- Gold answer: no
- Gold urbench_facts (audit):
  - USB flash drives first appeared on the market in late 2000.
  - The Godfather was released in 1972.
- truth: ____   relevance: ____

### D8-Xp #25 — qid `a0896de3fd13cd0f3e16`

- **Fact to check:** Atlantic cod is labelled vulnerable on the IUCN Red List of Threatened Species.
- Context from the sample row: titles_corpus: Atlantic cod
- Gold question_en (audit): Is Atlantic cod found in a vegemite sandwich?
- Gold answer: no
- Gold urbench_facts (audit):
  - Vegemite is a spread popular in Australia.
  - Vegemite is made from leftover brewers' yeast extract with various vegetable and spice additives. 
  - The Atlantic cod is found mostly in North America and Greenland.
- truth: ____   relevance: ____

### D8-Xp #26 — qid `557e389d7efe47c5bc4d`

- **Fact to check:** Blue has been used in art and decoration since ancient times.
- Context from the sample row: titles_corpus: Blue
- Gold question_en (audit): Is the most expensive color in the world Blue?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Blue is a primary color.
  - Blue is between violet and green on the visible light spectrum.
  - Lapis Lazuli is used to make ultramarine. 
  - Ultramarine is a pigment of Blue
  - Processing Lapis Lazuli into Ultramarine is the most expensive of color processes.
- truth: ____   relevance: ____

### D8-Xp #27 — qid `05e9cdc44f1b066badd7`

- **Fact to check:** Earliest depictions of moustaches trace back to Ancient Egypt's Old Kingdom era (ca. 2649–2130 B.C.).
- Context from the sample row: titles_corpus: Moustache
- Gold question_en (audit): Is it common for women to have moustaches?
- Gold answer: no
- Gold urbench_facts (audit):
  - Facial hair doesn't normally grow on women like it does on men.
  - A little bit of hair can grow between the upper lip and nose but it's a very small amount and generally not enough to be noticeable.
- truth: ____   relevance: ____

### D8-Xp #28 — qid `f4a336168f5165de7f0d`

- **Fact to check:** The risks of complications during or after anesthesia depend on the individual's health.
- Context from the sample row: titles_corpus: Anesthesia
- Gold question_en (audit): Is a northern fur seal needing emergency surgery in July likely a safe anesthesia candidate?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Northern fur seals fast throughout the mating season
  - It is recommended that patients, including animals, fast for a time before surgery that requires anesthesia 
  - Peak mating season for northern fur seals occurs in June and July
- truth: ____   relevance: ____

### D8-Xp #29 — qid `690a36d5fed5b17969aa`

- **Fact to check:** Jack Dempsey's career as a professional boxer lasted from 1914 to 1927.
- Context from the sample row: titles_corpus: Jack Dempsey
- Gold question_en (audit): Did Jack Dempsey fight the current WBC heavyweight champion?
- Gold answer: no
- Gold urbench_facts (audit):
  - Jack Dempsey died in 1983
  - The current WBC heavyweight champion is Tyson Fury
  - Tyson Fury was born in 1988
- truth: ____   relevance: ____

### D8-Xp #30 — qid `b3443bf3c312c9d144e4`

- **Fact to check:** The Panthéon was first used as a mausoleum in 1791.
- Context from the sample row: titles_corpus: Panthéon
- Gold question_en (audit): Is there a full Neptunian orbit between the first two burials of women in the Panthéon?
- Gold answer: no
- Gold urbench_facts (audit):
  - In 1907, Sophie Berthelot is the first woman to be interred in the Panthéon 
  - In 1995, Marie Curie is the second woman to be interred there
  - Neptune takes 165 years to go around the sun
- truth: ____   relevance: ____

## Sample D10-TF

Source: `outputs/efbpt/d10/student/tf_facts_sample30.jsonl (seed 8, n 30, job 109582; v1 teacher T1)`

### D10-TF #1 — qid `557e389d7efe47c5bc4d`

- **Fact to check:** Tyrian purple is a natural dye that was highly valued in antiquity for its rarity and expense.
- Context from the sample row: teacher entities: Ultramarine; Pigment; Lapis lazuli; Tyrian purple; Vantablack; YInMn blue
- Gold question_en (audit): Is the most expensive color in the world Blue?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Blue is a primary color.
  - Blue is between violet and green on the visible light spectrum.
  - Lapis Lazuli is used to make ultramarine. 
  - Ultramarine is a pigment of Blue
  - Processing Lapis Lazuli into Ultramarine is the most expensive of color processes.
- truth: ____   relevance: ____

### D10-TF #2 — qid `9576ab36999e196b87bb`

- **Fact to check:** The show is known for its portrayal of working-class life in London's East End.
- Context from the sample row: teacher entities: EastEnders; Water
- Gold question_en (audit): Is it possible to binge entire EastEnders series without water?
- Gold answer: no
- Gold urbench_facts (audit):
  - British TV series EastEnders has over 6,000 episodes as of 2020.
  - It would take approximately 125 days to binge watch the entire EastEnders TV series.
  - A human can last only 4 days without water.
- truth: ____   relevance: ____

### D10-TF #3 — qid `9711b3fc634e6f4f471d`

- **Fact to check:** World War II was a global conflict that lasted from 1939 to 1945.
- Context from the sample row: teacher entities: World War II; Alan Turing
- Gold question_en (audit): Would World War II have been the same without Alan Turing?
- Gold answer: no
- Gold urbench_facts (audit):
  - During WW2, the German Military used something called the Engima device to send messages secretly. 
  - Alan Turing broke the Enigma code, allowing German messages to be understood and intercepted.
- truth: ____   relevance: ____

### D10-TF #4 — qid `315575d5cbdd1d154744`

- **Fact to check:** Kurt Cobain was an American musician and artist, best known as the lead singer and guitarist of the grunge band Nirvana.
- Context from the sample row: teacher entities: Kurt Cobain; Project Sami Collen
- Gold question_en (audit): Would Kurt Cobain have benefited from Project Semicolon?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Project Semicolon is an American nonprofit organization known for its advocacy of mental health wellness and its focus as an anti-suicide initiative.
  - During the last years of his life, Cobain struggled with heroin addiction and chronic health problems such as depression.
  - Cobain died at the age of 27 from apparent suicide by gunshot.
- truth: ____   relevance: ____

### D10-TF #5 — qid `43da475d5c2db929c458`

- **Fact to check:** The Eiffel Tower is not designed to be submerged in water.
- Context from the sample row: teacher entities: Eiffel Tower; Arctic Ocean
- Gold question_en (audit): Could the Eiffel Tower be completely submerged at the Arctic Ocean's deepest point?
- Gold answer: yes
- Gold urbench_facts (audit):
  - The deepest point in the Arctic Ocean is 18,210 feet below the surface.
  - The Eiffel Tower is 1,063 feet tall.
- truth: ____   relevance: ____

### D10-TF #6 — qid `0d22526961c82ad6ef4a`

- **Fact to check:** Fairy tales often feature supernatural creatures, including fairies, and are frequently set in a fantasy world or a medieval setting.
- Context from the sample row: teacher entities: Fairy; Valkyrie; Fairy tale
- Gold question_en (audit): Is a fairy more prevalent in world myths than a valkyrie?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Valkyries are female figures that choose heroes to bring to Valhalla.
  - Valkyries are exclusive to Norse mythology.
  - A fairy is a mystical magical being that can be found in Celtic, Slavic, German, English, and French folklore.
- truth: ____   relevance: ____

### D10-TF #7 — qid `24a3aff40ef7fd04e208`

- **Fact to check:** Richard Dawkins is a British evolutionary biologist and science writer.
- Context from the sample row: teacher entities: Richard Dawkins; Secular humanism
- Gold question_en (audit): Would Richard Dawkins hypothetically refuse an offering of the Last rites?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Richard Dawkins is known as an outspoken atheist, well known for his criticism of creationism and intelligent design.
  - The Last rites, in Catholicism, are the last prayers and ministrations given to an individual of the faith, when possible, shortly before death.
- truth: ____   relevance: ____

### D10-TF #8 — qid `3486b6b5c69878bb088d`

- **Fact to check:** The United States Department of Agriculture (USDA) tracks data on cabbage production by state.
- Context from the sample row: teacher entities: Arizona; California; Cabbage
- Gold question_en (audit): Do more Cauliflower grow in Arizona than California?
- Gold answer: no
- Gold urbench_facts (audit):
  - Cauliflower grows best in cool temperatures with lots of sun.
  - California is the largest producer of Cauliflower in the U.S.
  - Arizona has a hot arid climate.
- truth: ____   relevance: ____

### D10-TF #9 — qid `63fb8f40ca6c2226dd01`

- **Fact to check:** Dr. Seuss was an American author, poet, and illustrator best known for his children's books.
- Context from the sample row: teacher entities: Helen Palmer Geisel; Seuss
- Gold question_en (audit): Did the death of Helen Palmer have a significant effect on Dr. Seuss?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Dr. Seuss's real name was Theodor Geisel.
  - Theodor Geisel was married to Helen Palmer at the time of her suicide.
  - Theodor Geisel is quoted having said he considered suicide after the death of his wife.
- truth: ____   relevance: ____

### D10-TF #10 — qid `d47120efd0f09badd848`

- **Fact to check:** Peppermint flavoring is often used in holiday treats and drinks during the Christmas season.
- Context from the sample row: teacher entities: Menthol; Christmas
- Gold question_en (audit): Is menthol associated with Christmas?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Menthol is the chemical in mint products that give mint its characteristic cool and tangy taste.
  - Peppermint is a popular candy flavor during Christmas season.
- truth: ____   relevance: ____

### D10-TF #11 — qid `48da75d87c66754ccc2e`

- **Fact to check:** As of 1941, the Speaker of the United States House of Representatives was Sam Rayburn.
- Context from the sample row: teacher entities: Pearl Harbor; Speaker of the United States House of Representatives
- Gold question_en (audit): Was only woman to serve as U.S. Speaker of the House alive during the attack on Pearl Harbor?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Nancy Pelosi is the only woman to ever serve as Speaker of the United States House of Representatives.
  - Nancy Pelosi was born on Mar 26, 1940
  - The attach on Pearl Harbor occurred on December 7, 1941
- truth: ____   relevance: ____

### D10-TF #12 — qid `9fee16a5e703c2fdcd33`

- **Fact to check:** Blair White's views on social issues have been criticized by some conservatives.
- Context from the sample row: teacher entities: Blair White; Traditionalist conservatism
- Gold question_en (audit): Does conservatism repulse Blaire White?
- Gold answer: no
- Gold urbench_facts (audit):
  - Blaire White is a Youtuber.
  - Blaire White posts content that leans towards conservative politics.
- truth: ____   relevance: ____

### D10-TF #13 — qid `06ebd3ca1fe98d5796f7`

- **Fact to check:** Email is a method of exchanging digital messages from an sender to one or more recipients.
- Context from the sample row: teacher entities: Douglas Adams; Email
- Gold question_en (audit): Did Douglas Adams use email as a child?
- Gold answer: no
- Gold urbench_facts (audit):
  - Douglas Adams was born in 1952.
  - Modern email did not emerge until 1977.
- truth: ____   relevance: ____

### D10-TF #14 — qid `bcd77b200a2b90ca2ecc`

- **Fact to check:** Pollock's paintings are characterized by intricate webs of lines and patterns.
- Context from the sample row: teacher entities: Bloodstain pattern analysis; Jackson Pollock
- Gold question_en (audit): Is it understandable to compare a blood spatter pattern to a Jackson Pollock piece?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Jackson Pollock is well known for a style of art formed through splashing liquids on canvas.
  - Blood spatter patterns are caused by a splash of blood onto a surface or multiple surfaces.
- truth: ____   relevance: ____

### D10-TF #15 — qid `cb9eae3948f28e236477`

- **Fact to check:** Giant pandas are members of the Ursidae family, which includes bears.
- Context from the sample row: teacher entities: Giant Panda; Gorilla; Bear
- Gold question_en (audit): Do gorillas fight with panda bears?
- Gold answer: no
- Gold urbench_facts (audit):
  - Gorilla distribution is limited to areas of the continent of Africa
  - Panda bears are found in the wild only in China
- truth: ____   relevance: ____

### D10-TF #16 — qid `b77170e15c0b1bd72340`

- **Fact to check:** Durian is banned in many public places in Southeast Asia due to its strong odor.
- Context from the sample row: teacher entities: Durian; Fruit
- Gold question_en (audit): Are Durian fruits an olfactory delight?
- Gold answer: no
- Gold urbench_facts (audit):
  - Durian is a plant type that produces several kinds of fruit.
  - Olfactory refers to the human sense of smell.
  - Pleasant smells according to polls include flowers and sweet foods.
  - Durian fruits have been banned in Singapore due to its overwhelming smell.
- truth: ____   relevance: ____

### D10-TF #17 — qid `9bb59488a227a5f5cd62`

- **Fact to check:** GB is an abbreviation for gigabyte, a unit of digital information.
- Context from the sample row: teacher entities: Game of Thrones; GB; HD video
- Gold question_en (audit): Can you save every HD episode of Game of Thrones on Samsung Galaxy A10e?
- Gold answer: no
- Gold urbench_facts (audit):
  - The Samsung Galaxy A10e has 32GB of storage.
  - The average storage requirement of an HD episode of Game of Thrones is 600MB
  - There are 60 total episodes of Game of Thrones.
  - There are 1000MB in one GB.
- truth: ____   relevance: ____

### D10-TF #18 — qid `d0183768701c74f966c5`

- **Fact to check:** Hindus often visit sacred sites to perform rituals and seek blessings from their deities.
- Context from the sample row: teacher entities: Shiva; Hindu pilgrimage sites
- Gold question_en (audit): Do worshipers of Shiva make a pilgrimage to the Holy Land?
- Gold answer: no
- Gold urbench_facts (audit):
  - The Holy Land is sacred to Judaism, Islam and Christianity
  - Worshipers of Shiva are adherents of Hinduism
- truth: ____   relevance: ____

### D10-TF #19 — qid `fba624d8c01833419760`

- **Fact to check:** The Northern Hemisphere experiences winter while the Southern Hemisphere experiences summer during the months of December, January, and February.
- Context from the sample row: teacher entities: August; Season; Southern Hemisphere; Northern Hemisphere
- Gold question_en (audit): Is August a winter month for part of the world?
- Gold answer: yes
- Gold urbench_facts (audit):
  - August is a summer month in the northern hemisphere.
  - However, the seasons are opposite south of the Equator.
  - August is in the middle of winter for Australia, Antarctica, and parts of Africa and South America.
- truth: ____   relevance: ____

### D10-TF #20 — qid `43da475d5c2db929c458`

- **Fact to check:** The Arctic Ocean's deepest point is in the Molloy Deep, with a depth of approximately 5,370 meters (17,650 feet).
- Context from the sample row: teacher entities: Eiffel Tower; Arctic Ocean
- Gold question_en (audit): Could the Eiffel Tower be completely submerged at the Arctic Ocean's deepest point?
- Gold answer: yes
- Gold urbench_facts (audit):
  - The deepest point in the Arctic Ocean is 18,210 feet below the surface.
  - The Eiffel Tower is 1,063 feet tall.
- truth: ____   relevance: ____

### D10-TF #21 — qid `a0896de3fd13cd0f3e16`

- **Fact to check:** A sandwich is a food item typically consisting of two or more slices of bread with fillings between them.
- Context from the sample row: teacher entities: Atlantic cod; Cod as food; Sandwich
- Gold question_en (audit): Is Atlantic cod found in a vegemite sandwich?
- Gold answer: no
- Gold urbench_facts (audit):
  - Vegemite is a spread popular in Australia.
  - Vegemite is made from leftover brewers' yeast extract with various vegetable and spice additives. 
  - The Atlantic cod is found mostly in North America and Greenland.
- truth: ____   relevance: ____

### D10-TF #22 — qid `25a088d9d2ce674e639a`

- **Fact to check:** Jackfruit has been reported to cause allergic reactions in some individuals, including anaphylaxis.
- Context from the sample row: teacher entities: Jackfruit; Food allergy; Anaphylaxis
- Gold question_en (audit): Can eating grapefruit kill besides allergies or choking?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Grapefruit is a citrus fruit consumed mostly during the summer months.
  - Chemicals in grapefruit can interact with medications such as statins.
  - Grapefruit can lead to too much absorption of statin medicine.
  - Too much statins can cause severe muscle pain, liver damage, kidney failure and death. 
- truth: ____   relevance: ____

### D10-TF #23 — qid `cac8e6300a98cf6128af`

- **Fact to check:** Numerology is the belief in the mystical relationship between numbers and their influence on human life.
- Context from the sample row: teacher entities: 19th century; President of the United States; Hong Kong; Numerology
- Gold question_en (audit): Is number of different US President's in 1800s a lucky number in Hong Kong?
- Gold answer: no
- Gold urbench_facts (audit):
  - There were 24 different US President's in the 1800s.
  - 4 is an unlucky number in Chinese numerology.
  - Where East Asian and Western cultures blend, such as in Hong Kong, it is possible in some buildings that the thirteenth floor along with all the floors with 4s to be omitted. 
- truth: ____   relevance: ____

### D10-TF #24 — qid `593e66cbdec8e3a15852`

- **Fact to check:** The number of letters in the Ancient Greek alphabet has remained relatively consistent throughout history.
- Context from the sample row: teacher entities: Survivor (TV series); Ancient Greek alphabet
- Gold question_en (audit): Are seasons of Survivor surpassed by number of Ancient Greek letters?
- Gold answer: no
- Gold urbench_facts (audit):
  - The reality show Survivor has aired 40 seasons as of 2020.
  - The Ancient Greek alphabet consisted of 24 letters.
- truth: ____   relevance: ____

### D10-TF #25 — qid `064d8faaa8f164b55270`

- **Fact to check:** Unicycles require balance and coordination to operate.
- Context from the sample row: teacher entities: Apollo 15; Unicycle
- Gold question_en (audit): Would the crew of Apollo 15 have difficulty riding a unicycle?
- Gold answer: yes
- Gold urbench_facts (audit):
  - There were 3 astronauts in the crew of the Apollo 15 mission.
  - A unicycle only contains one saddle, and is typically only operated by a single person.
- truth: ____   relevance: ____

### D10-TF #26 — qid `6f4c8a2789c0305c4f63`

- **Fact to check:** Maya Angelou was an American poet, memoirist, and civil rights activist.
- Context from the sample row: teacher entities: Maya Angelou; Children of Maya Angelou
- Gold question_en (audit): Did any of Maya Angelou's children follow in her footsteps?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Maya Angelou was a civil rights activist and author.
  - Maya Angelou had a son named Guy Johnson in 1945.
  - Guy Johnson is an author that has written over twenty books and essays.
  - Guy Johnson's books explore many civil rights themes.
- truth: ____   relevance: ____

### D10-TF #27 — qid `dce3695a2aaa8d7dba5a`

- **Fact to check:** Bill Nye was born on November 27, 1955.
- Context from the sample row: teacher entities: Bill Nye; Franklin D. Roosevelt
- Gold question_en (audit): Did Bill Nye vote for Franklin Delano Roosevelt?
- Gold answer: no
- Gold urbench_facts (audit):
  - Bill Nye was born in 1955
  - Franklin Delano Roosevelt's last election was in 1944
- truth: ____   relevance: ____

### D10-TF #28 — qid `a406e06cd0164c1c6d15`

- **Fact to check:** Auburn, New York is a city located in Cayuga County, New York, United States.
- Context from the sample row: teacher entities: Auburn, New York; September; Tree; Fall foliage
- Gold question_en (audit): Would the trees in Auburn, New York be changing colors in September?
- Gold answer: yes
- Gold urbench_facts (audit):
  - In New York, fall begins during the month of September. 
  - When the weather gets cooler in the fall, leaves begin to change colors. 
- truth: ____   relevance: ____

### D10-TF #29 — qid `c70d40b57fa67ea13ed6`

- **Fact to check:** The appearance of a person's hair can be influenced by their genetics, hormones, and environmental factors.
- Context from the sample row: teacher entities: Tonsure; Hair
- Gold question_en (audit): Does a person using tonsure have hair at the top of their scalp?
- Gold answer: no
- Gold urbench_facts (audit):
  - Tonsure involves shaving some or all of the hair from the head.
  - Tonsure styles include a large bald spot at the top of the scalp.
- truth: ____   relevance: ____

### D10-TF #30 — qid `9850eb1534206facd067`

- **Fact to check:** A microchip is a miniaturized electronic device that contains a set of electronic circuits on a small flat piece of semiconductor material.
- Context from the sample row: teacher entities: Central processing unit; Integrated circuit; Microchip
- Gold question_en (audit): Would a modern central processing unit circuit chip fit on a housekey?
- Gold answer: no
- Gold urbench_facts (audit):
  - A CPU circuit chip is about an inch across.
  - A housekey is generally less than a half-inch across.
- truth: ____   relevance: ____

## Sample D10v2-TF

Source: `outputs/efbpt/d10v2/student/tf_facts_sample30.jsonl (seed 8, n 30, job 110655; v2 teacher T1-v2)`

### D10v2-TF #1 — qid `4bbf4b169b4981ad5a34`

- **Fact to check:** The distance between New York City and Miami is approximately 1,280 miles (2,060 km).
- Context from the sample row: teacher entities: New York; Miami; Human swimming
- Gold question_en (audit): Can you swim to Miami from New York?
- Gold answer: no
- Gold urbench_facts (audit):
  - The longest distance swam by a person is 139.8 miles.
  - It is over 1,000 miles from New York to Miami.
- truth: ____   relevance: ____

### D10v2-TF #2 — qid `8d06b619a7045ed02f51`

- **Fact to check:** The song 'Diamonds Are Forever' was released in 1971 and performed by Shirley Bassey.
- Context from the sample row: teacher entities: Shirley Bassey; Diamonds (Shirley Bassey and Diamonds album)
- Gold question_en (audit): Is the title of Shirley Bassey's 1971 diamond song a true statement?
- Gold answer: no
- Gold urbench_facts (audit):
  - Shirley Bassey recorded the song Diamonds are Forever in 1971,
  - Over time, diamonds degrade and turn into graphite.
  - Graphite is the same chemical composition found in pencils.
- truth: ____   relevance: ____

### D10v2-TF #3 — qid `8d3ddaee20ad48edc066`

- **Fact to check:** Helio Gracie had seven children.
- Context from the sample row: teacher entities: Brazilian Jiu-Jitsu; Gracie family
- Gold question_en (audit): Did Brazilian jiu-jitsu Gracie founders have at least a baker's dozen of kids between them?
- Gold answer: yes
- Gold urbench_facts (audit):
  - A baker's dozen refers to 13 of anything.
  - Brazilian jiu-jitsu was founded by Carlos and Helio Gracie.  
  - Helio Gracie had 9 children.
  - Carlos Gracie had 11 children.
- truth: ____   relevance: ____

### D10v2-TF #4 — qid `2e508efbf0b72f1af2c2`

- **Fact to check:** Oxygen is essential for human respiration and survival.
- Context from the sample row: teacher entities: Quartz; Oxygen; Plant
- Gold question_en (audit): Could Quartz be useful to humans if plants died off and there was no oxygen?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Plants produce oxygen which is needed by humans to survive.
  - Quartz is a hard mineral substance made of several elements.
  - Quartz is composed of silicon and oxygen.
  - Quartz can be melted at high temperatures.
- truth: ____   relevance: ____

### D10v2-TF #5 — qid `3f5af97fdff20028347b`

- **Fact to check:** Regular high-definition television broadcasts did not begin until the late 20th century.
- Context from the sample row: teacher entities: French Revolution; Television
- Gold question_en (audit): Was the French Revolution televised?
- Gold answer: no
- Gold urbench_facts (audit):
  - The french revolution occurred during the 1700's.
  - Television was invented in 1927. 
- truth: ____   relevance: ____

### D10v2-TF #6 — qid `0d22526961c82ad6ef4a`

- **Fact to check:** In Norse mythology, a Valkyrie is a female figure who chooses which soldiers die in battle and take them to Valhalla.
- Context from the sample row: teacher entities: Valkyrie; Norse mythology; Fairy tale; Fairy
- Gold question_en (audit): Is a fairy more prevalent in world myths than a valkyrie?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Valkyries are female figures that choose heroes to bring to Valhalla.
  - Valkyries are exclusive to Norse mythology.
  - A fairy is a mystical magical being that can be found in Celtic, Slavic, German, English, and French folklore.
- truth: ____   relevance: ____

### D10v2-TF #7 — qid `20693ee66ad7ca890e08`

- **Fact to check:** The original exterior of the pyramids was often covered in hieroglyphics and other inscriptions.
- Context from the sample row: teacher entities: Egyptian pyramids
- Gold question_en (audit): Do the Egyptian pyramids look the same from outside as they did when new?
- Gold answer: no
- Gold urbench_facts (audit):
  - When originally built, the Great Pyramids had a thin surface of limestone covering the bricks, making them shine in the sun.
  - Over the centuries, the limestone layer has been removed by thieves and erosion, exposing the more common stone bricks underneath.
- truth: ____   relevance: ____

### D10v2-TF #8 — qid `30fcf62f58849cb4ab05`

- **Fact to check:** The Great Pyramid of Giza is believed to have been built as a tomb for Pharaoh Khufu.
- Context from the sample row: teacher entities: Great Pyramid of Giza; Ancient Egypt; Theocracy
- Gold question_en (audit): Were the Great Pyramids built by a theocratic government?
- Gold answer: yes
- Gold urbench_facts (audit):
  - The Great Pyramids were built by the Ancient Egyptians.
  - A theocracy is a government that is led by religious leaders or who worships their leader as a god.
  - The Pharaohs of Ancient Egypt were worshipped as children of the Sun god Ra.
- truth: ____   relevance: ____

### D10v2-TF #9 — qid `5897ec22db850f7b416e`

- **Fact to check:** John Key was the 38th Prime Minister of New Zealand.
- Context from the sample row: teacher entities: John Key; Prime Minister of New Zealand
- Gold question_en (audit): As of 2020 have more women succeeded John Key than preceded him?
- Gold answer: no
- Gold urbench_facts (audit):
  - John Key, the 38th Prime Minister of New Zealand, has had one woman succeed him as Prime Minister.
  - John key was preceded by two women as Prime Minister of New Zealand.
- truth: ____   relevance: ____

### D10v2-TF #10 — qid `b77170e15c0b1bd72340`

- **Fact to check:** Durian is a tropical fruit native to Southeast Asia.
- Context from the sample row: teacher entities: Durian
- Gold question_en (audit): Are Durian fruits an olfactory delight?
- Gold answer: no
- Gold urbench_facts (audit):
  - Durian is a plant type that produces several kinds of fruit.
  - Olfactory refers to the human sense of smell.
  - Pleasant smells according to polls include flowers and sweet foods.
  - Durian fruits have been banned in Singapore due to its overwhelming smell.
- truth: ____   relevance: ____

### D10v2-TF #11 — qid `4589898e4852e5389728`

- **Fact to check:** The YMCA's headquarters is located in Geneva, Switzerland.
- Context from the sample row: teacher entities: YMCA
- Gold question_en (audit): Can you get Raclette in YMCA headquarters city?
- Gold answer: yes
- Gold urbench_facts (audit):
  - YMCA is headquartered in Geneva, Switzerland.
  - Raclette is a melted cheese and potato dish.
  - Raclette is one of several foods Geneva, Switzerland is famous for.
- truth: ____   relevance: ____

### D10v2-TF #12 — qid `91917c5e30dbf8552b34`

- **Fact to check:** Paella is a traditional Spanish dish that originated in the Valencia region.
- Context from the sample row: teacher entities: Paella; Veganism
- Gold question_en (audit): Would a vegan eat a traditional Paella dish?
- Gold answer: no
- Gold urbench_facts (audit):
  - Vegans do not consume animals or products derived from animals.
  - The traditional Paella recipe includes rabbit and chicken as two of the ingredients.
- truth: ____   relevance: ____

### D10v2-TF #13 — qid `f7a32d14efd589015fe1`

- **Fact to check:** The herpes simplex virus is a human pathogen that can survive on surfaces for a short period of time.
- Context from the sample row: teacher entities: Herpes simplex virus; Venus
- Gold question_en (audit): Can Herpes simplex virus spread on Venus?
- Gold answer: no
- Gold urbench_facts (audit):
  - Herpes simplex virus is a disease that has the structure of a tiny protein cage.
  - Venus is the hottest planet and its temperature can reach 900°F.
  - Proteins lose their structure and break down at temperatures above 105.8°F.
- truth: ____   relevance: ____

### D10v2-TF #14 — qid `069c88860654c9c5df0b`

- **Fact to check:** Tennis is a sport played between two players using rackets to hit a ball over a net.
- Context from the sample row: teacher entities: Sea Turtle; Tennis Racket; Tennis
- Gold question_en (audit): Can a sea turtle play tennis using a tennis racket?
- Gold answer: no
- Gold urbench_facts (audit):
  - to play tennis, a human-like hand is needed to properly hold the tennis racket
  - sea turtles have flippers and not human-like hands
- truth: ____   relevance: ____

### D10v2-TF #15 — qid `a54b16376f82a6e14bf9`

- **Fact to check:** The density of a stone can vary depending on its composition, but it is typically around 2-3 grams per cubic centimeter.
- Context from the sample row: teacher entities: Earth's atmosphere; Stone; Density; Buoyancy
- Gold question_en (audit): Will a rock float in the atmosphere of Earth?
- Gold answer: no
- Gold urbench_facts (audit):
  - Things only float if they are less dense than the surrounding material.
  - Rocks are denser than air.
  - The atmosphere of Earth is made up of air.
- truth: ____   relevance: ____

### D10v2-TF #16 — qid `ae0891799990442fbb0e`

- **Fact to check:** Paul Bunyan is often depicted in rural or wilderness settings.
- Context from the sample row: teacher entities: Paul Bunyan; Urban planning
- Gold question_en (audit): Would Paul Bunyan hypothetically be a poor choice for an urban planner?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Paul Bunyan was a legendary giant from tall tale stories.
  - Urban planners need to design buildings and ceilings in cities.
  - Paul Bunyan lived in the country side with a giant blue ox.
  - New York city apartment ceilings average around 8 feet in height.
  - Paul Bunyan was over seven feet tall.
- truth: ____   relevance: ____

### D10v2-TF #17 — qid `a406e06cd0164c1c6d15`

- **Fact to check:** The change in leaf color for trees in the northeastern United States is triggered by the shortening of daylight hours and the cooling of temperatures.
- Context from the sample row: teacher entities: Auburn, New York; September; Fall foliage
- Gold question_en (audit): Would the trees in Auburn, New York be changing colors in September?
- Gold answer: yes
- Gold urbench_facts (audit):
  - In New York, fall begins during the month of September. 
  - When the weather gets cooler in the fall, leaves begin to change colors. 
- truth: ____   relevance: ____

### D10v2-TF #18 — qid `9052063382d30870a2ed`

- **Fact to check:** Sertraline is an antidepressant medication.
- Context from the sample row: teacher entities: Sertraline; Depression
- Gold question_en (audit): Would Seroquel be the first treatment recommended by a doctor to someone with depression?
- Gold answer: no
- Gold urbench_facts (audit):
  - Seroquel is a powerful drug that is prescribed for bipolar disorder.
  - Seroquel has sedating effects and can increase feelings of depression.
  - Depression is usually treated by SSRI's. 
  - Seroquel is an atypical antipsychotic.
- truth: ____   relevance: ____

### D10v2-TF #19 — qid `b3443bf3c312c9d144e4`

- **Fact to check:** Neptune takes approximately 165 Earth years to complete one orbit around the Sun.
- Context from the sample row: teacher entities: Panthéon, Paris; Neptune
- Gold question_en (audit): Is there a full Neptunian orbit between the first two burials of women in the Panthéon?
- Gold answer: no
- Gold urbench_facts (audit):
  - In 1907, Sophie Berthelot is the first woman to be interred in the Panthéon 
  - In 1995, Marie Curie is the second woman to be interred there
  - Neptune takes 165 years to go around the sun
- truth: ____   relevance: ____

### D10v2-TF #20 — qid `d15a33b8f88dc5ce32b0`

- **Fact to check:** Some musicians modify their instruments to suit their playing style or musical genre.
- Context from the sample row: teacher entities: Dholki
- Gold question_en (audit): Do drummers need spare strings?
- Gold answer: no
- Gold urbench_facts (audit):
  - Drummers usually work with guitarists or other stringed instrumentalists.
  - However, drum sets do not contain strings.
  - Musicians usually change their own strings as necessary, so it is their band mates, not the drummer, who carries the spare strings.
- truth: ____   relevance: ____

### D10v2-TF #21 — qid `3f5af97fdff20028347b`

- **Fact to check:** The first public demonstration of a television system was held by John Logie Baird in 1926.
- Context from the sample row: teacher entities: French Revolution; Television
- Gold question_en (audit): Was the French Revolution televised?
- Gold answer: no
- Gold urbench_facts (audit):
  - The french revolution occurred during the 1700's.
  - Television was invented in 1927. 
- truth: ____   relevance: ____

### D10v2-TF #22 — qid `9247b5afb2eef4c9f95b`

- **Fact to check:** The tibia, also known as the shinbone, is a bone in the lower leg of humans and other animals.
- Context from the sample row: teacher entities: Minotaur; Tibia; Football
- Gold question_en (audit): Can a minotaur hypothetically injure a tibia playing football?
- Gold answer: yes
- Gold urbench_facts (audit):
  - A minotaur is a mythological creature with the head of a bull and the body of a human.
  - The tibia is the bone between the knee and the ankle in humans.
  - Tibia injuries are common injuries in contact sports such as football.
- truth: ____   relevance: ____

### D10v2-TF #23 — qid `20abc7ba538a2fd1dd76`

- **Fact to check:** Dietary fiber is a type of carbohydrate that is not easily broken down by the body.
- Context from the sample row: teacher entities: Small intestine; Dietary fiber
- Gold question_en (audit): Will the small intenstine break down a cotton ball?
- Gold answer: no
- Gold urbench_facts (audit):
  - The small intestine does not digest cellulose 
  - Cotton fiber is 90% cellulose
- truth: ____   relevance: ____

### D10v2-TF #24 — qid `acb7edac010554509519`

- **Fact to check:** There are alternative sources of food and oxygen for humans, such as algae and some types of bacteria, but they are not sufficient to support the entire human population.
- Context from the sample row: teacher entities: Chlorophyll; Human; Ecosystem; Food chain; Plant
- Gold question_en (audit): Would human race go extinct without chlorophyll?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Chlorophyll is a pigment in plants responsible for photosynthesis.
  - Photosynthesis is the process by which plants release oxygen into the atmosphere.
  - Humans need oxygen to live.
- truth: ____   relevance: ____

### D10v2-TF #25 — qid `4e1b65e81ec09397b26e`

- **Fact to check:** Giant squid are found in the deep waters of the Atlantic, Pacific, and Indian Oceans.
- Context from the sample row: teacher entities: Giant Squid; Deep-Sea Environment; Diving Equipment; Human Diver
- Gold question_en (audit): Is capturing giant squid in natural habitat impossible with no gear?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Giant squids live between 1,000 and 3,800 feet in the ocean.
  - With a dry dive suit, a scuba tank, gloves, and so on, divers can reach depths of around 1000 feet.
  - Without scuba gear people can safely dive a maximum of 60 feet without feeling the negative risks associated with diving beyond the limit.
- truth: ____   relevance: ____

### D10v2-TF #26 — qid `064d8faaa8f164b55270`

- **Fact to check:** The Apollo 15 crew consisted of David Scott, James Irwin, and Alfred Worden.
- Context from the sample row: teacher entities: Apollo 15; Unicycle
- Gold question_en (audit): Would the crew of Apollo 15 have difficulty riding a unicycle?
- Gold answer: yes
- Gold urbench_facts (audit):
  - There were 3 astronauts in the crew of the Apollo 15 mission.
  - A unicycle only contains one saddle, and is typically only operated by a single person.
- truth: ____   relevance: ____

### D10v2-TF #27 — qid `63fb8f40ca6c2226dd01`

- **Fact to check:** Helen Palmer Geisel was the wife of Theodor Geisel, better known as Dr. Seuss.
- Context from the sample row: teacher entities: Helen Palmer Geisel; Theodor Geisel
- Gold question_en (audit): Did the death of Helen Palmer have a significant effect on Dr. Seuss?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Dr. Seuss's real name was Theodor Geisel.
  - Theodor Geisel was married to Helen Palmer at the time of her suicide.
  - Theodor Geisel is quoted having said he considered suicide after the death of his wife.
- truth: ____   relevance: ____

### D10v2-TF #28 — qid `bd734e6cb6674e25dbe8`

- **Fact to check:** The Church of Scientology has a policy of disconnection with those who criticize it.
- Context from the sample row: teacher entities: Tom Cruise; L. Ron Hubbard; Church of Scientology
- Gold question_en (audit): Would Tom Cruise ever insult L. Ron Hubbard?
- Gold answer: no
- Gold urbench_facts (audit):
  - Tom Cruise is an outspoken advocate for the Church of Scientology and its associated social programs.
  - The Church of Scientology was founded by L. Ron Hubbard.
  - L. Ron Hubbard is a revered and god-like figure in The Church of Scientology.
- truth: ____   relevance: ____

### D10v2-TF #29 — qid `9384dfc0aabc73026925`

- **Fact to check:** The human heart is a muscular organ that pumps blood throughout the body.
- Context from the sample row: teacher entities: Human Heart; Symbolism of the Heart
- Gold question_en (audit): Do anatomical and symbolic hearts look remarkably different?
- Gold answer: yes
- Gold urbench_facts (audit):
  - Symbolic hearts are sharply pointed at the bottom and feature a sharp valley between the bumps at the top.
  - Anatomical hearts are rounded, have numerous vascular tubes entering and exiting them, and do not feature sharp angles.
- truth: ____   relevance: ____

### D10v2-TF #30 — qid `a8bbf2e84d0649b1c52c`

- **Fact to check:** Some monastic orders have rules that restrict the activities of their members.
- Context from the sample row: teacher entities: Monk; War
- Gold question_en (audit): Are monks forbidden from engaging in warfare?
- Gold answer: no
- Gold urbench_facts (audit):
  - Monks are members of religious orders that usually take vows of poverty, chastity, and obedience.
  - The Knights Templar were a religious order that fought during the Crusades and captured Jerusalem in 1099.
  - Buddhist Shaolin monks developed very powerful martial arts skills, have defended temples during conquests.
- truth: ____   relevance: ____

