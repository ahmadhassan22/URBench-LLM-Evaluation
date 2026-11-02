# D10 — TF vs A_r2: the 28 losses and 38 wins, row by row

Read-only. Built from the sealed files only: `outputs/efbpt/d10/student/d10_armA_r2.jsonl`, `d10_armTF.jsonl`, `outputs/efbpt/d10/teacher/t_facts.jsonl`, and `R4_rows.b_all_rows` of `outputs/efbpt/d10/student/d10_summary.json` (gold evidence titles after norm(), teacher entities, hits). Companion of `docs/handoff/D10_FULL_READOUT.md` §6.

Conventions: a **loss** is a row where A_r2 is right and TF is wrong; a **win** is the reverse. "R4 hit" = at least one gold required title appears (after norm()) in the teacher's parsed entities (`hits` non-empty). Preds: yes / no / — (unparsed = wrong). Facts are the post-T2 lists exactly as injected (V8 PASS). The Urdu question is copied from the sealed A_r2 prompt.

## A. The 28 TF losses (A_r2 right, TF wrong)

By (gold, TF pred): gold yes → TF no: 17; gold no → TF yes: 10; gold no → TF unparsed: 1. R4 hit yes: 22, no: 6.

### 1. `01c27885cede6741bc53` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا سپرنیچرل نے 2001 میں دی سی ڈبلیو پر شروع ہونے والے شو کے سیزنز کا ریکارڈ توڑا؟
- Gold evidence titles (R4_rows, normalised): one tree hill (tv series); supernatural (american tv series); supernatural (season 1); the cw; the wb
- R4 hits: the cw
- Teacher entities: Supernatural (TV series); The CW; 2001
- Teacher facts (6):
  - Supernatural is an American fantasy horror television series created by Eric Kripke.
  - The CW is an American English-language television network that was formed as a result of the merger of The WB and UPN.
  - The WB and UPN merged to form The CW in 2006.
  - Supernatural premiered on September 13, 2005, on The WB, and later became part of The CW's lineup.
  - The CW began operations on September 18, 2006.
  - Supernatural ran for 15 seasons, concluding on November 19, 2020.

### 2. `15885efe6f91724c16d9` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا آوڈی آر ایٹ وی-ٹین پلس کے لیے صوتی رکاوٹ توڑنا ممکن نہیں؟
- Gold evidence titles (R4_rows, normalised): audi r8; audi r8 (type 4s); audi s8; sound barrier; speed of sound
- R4 hits: audi r8; sound barrier
- Teacher entities: Audi R8; Sound barrier
- Teacher facts (6):
  - The Audi R8 is a mid-engined, 2-seater sports car, which uses Audi's trademark quattro permanent all-wheel drive system.
  - The Audi R8 V10 Plus is a high-performance variant of the Audi R8.
  - The sound barrier, also known as sonic boom or supersonic boom, is the sudden, sharp noise that occurs as an object breaks the sound barrier.
  - The sound barrier is broken when an object travels at a speed greater than the speed of sound, approximately 768 mph (1,236 km/h) at sea level in dry air at a temperature of 59 °F (15 °C).
  - The Audi R8 V10 Plus has a top speed of 199 mph (320 km/h).
  - The speed of sound varies depending on atmospheric conditions, such as temperature, humidity, and air pressure.

### 3. `2a4e1d4524997b6263bf` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا آئرس (اساطیر) اور ہرمیز فرضی طور پر یو پی ایس کی ملازمت میں جدوجہد کریں گے؟
- Gold evidence titles (R4_rows, normalised): hermes; iris (mythology); united parcel service
- R4 hits: hermes; iris (mythology)
- Teacher entities: Iris (mythology); Hermes; UPS
- Teacher facts (6):
  - In Greek mythology, Iris is the goddess of the rainbow and a messenger of the gods.
  - Hermes is the Greek messenger god and the patron of commerce, thieves, and travelers.
  - UPS is an American multinational package delivery and supply chain management company.
  - In some versions of Greek mythology, Iris and Hermes are depicted as having a close relationship.
  - Hermes is often associated with speed and swiftness, being the messenger of the gods.
  - UPS is known for its logistics and delivery services, employing a large number of people worldwide.

### 4. `2e8e9e51f0f290e32d27` — gold **yes**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا کسی ثقافت نے اجوائن کو موت سے منسلک کیا؟
- Gold evidence titles (R4_rows, normalised): celery
- R4 hits: (none)
- Teacher entities: Ajwain; Culture; Death
- Teacher facts (6):
  - Ajwain is a herb used in various culinary and medicinal traditions.
  - Ajwain seeds have been used for centuries in Ayurvedic medicine.
  - In some cultures, certain herbs and spices are associated with spiritual or mystical properties.
  - Death is a universal human experience that has been interpreted and symbolized in many ways across cultures.
  - In many cultures, certain plants or flowers are used in funeral rites or as symbols of mourning.
  - The symbolic meanings of plants and herbs can vary greatly between cultures and historical periods.

### 5. `315575d5cbdd1d154744` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا کرٹ کوبائن کو پروجیکٹ سیمی کولن سے فائدہ ہوتا؟
- Gold evidence titles (R4_rows, normalised): kurt cobain; project semicolon; suicide of kurt cobain
- R4 hits: kurt cobain
- Teacher entities: Kurt Cobain; Project Sami Collen
- Teacher facts (6):
  - Kurt Cobain was an American musician and artist, best known as the lead singer and guitarist of the grunge band Nirvana.
  - Project Sami Collen is not a widely recognized or documented project.
  - Kurt Cobain struggled with depression and anxiety throughout his life.
  - Kurt Cobain passed away on April 5, 1994.
  - There is no widely available information about Kurt Cobain's involvement in a project called Project Sami Collen.
  - Kurt Cobain's death was a significant event in the music industry and had a lasting impact on his fans and fellow musicians.

### 6. `3570d57f16a40488ff42` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا سلیکون کے سوٹ جودو کو مشکل بناتے ہیں؟
- Gold evidence titles (R4_rows, normalised): judo; keikogi; leopold's maneuvers; silicone; silicone rubber
- R4 hits: judo
- Teacher entities: Silicon; Judo
- Teacher facts (6):
  - Silicon is a chemical element with the symbol Si and atomic number 14.
  - Silicon is used in various forms in the manufacture of protective gear and clothing, including suits.
  - Judo is a Japanese martial art that focuses on grappling and throwing techniques.
  - Judo is an Olympic sport and is practiced worldwide.
  - Judo competitions, or matches, are won by throwing or takedowns, or by submission.
  - Protective gear, including suits, is often worn in judo to prevent injury.

### 7. `38ef97eb7cdd4200fd00` — gold **yes**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا سنہری بالوں والی سبز آنکھوں والی سارہ پیکسٹن کو لاطینی سمجھا جاتا ہے؟
- Gold evidence titles (R4_rows, normalised): latin america; latino (demonym); mexico; sara paxton
- R4 hits: (none)
- Teacher entities: Sarah Paxton; Latin Americans
- Teacher facts (6):
  - Sarah Paxton is an American actress.
  - Sarah Paxton's film debut was in the 2004 film Sleepover.
  - Latin Americans are the citizens of the Latin American countries.
  - Latin American countries are generally considered to be those in Central America, South America, and the Caribbean where Romance languages are spoken.
  - The United States is not considered a Latin American country.
  - Sarah Paxton's ancestry is not publicly known to be Latin American.

### 8. `42c947475244ed02629e` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا سیپی کو گٹار بنانے میں استعمال کیا جا سکتا ہے؟
- Gold evidence titles (R4_rows, normalised): guitar; inlay (guitar); nacre; oyster
- R4 hits: guitar
- Teacher entities: Capybara; Guitar
- Teacher facts (6):
  - Capybaras are large rodents native to South America.
  - Capybaras have coarse, thin fur that is often used to make hats, gloves, and other clothing.
  - Guitars are stringed musical instruments typically made from wood.
  - Guitars can be made from a variety of woods, including spruce, mahogany, and rosewood.
  - Animal hides and skins are sometimes used to make musical instrument components, such as drumheads.
  - Capybara hide is not commonly used in musical instrument manufacturing.

### 9. `4589898e4852e5389728` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا آپ وائی ایم سی اے کے مرکزی دفتر کے شہر میں ریکلیٹ حاصل کر سکتے ہیں؟
- Gold evidence titles (R4_rows, normalised): lincoln park, chicago; raclette; swiss cheese union; swiss cuisine; ymca
- R4 hits: ymca
- Teacher entities: YMCA; Headquarters
- Teacher facts (6):
  - YMCA is a worldwide youth organization.
  - The YMCA was founded in 1844 in London, England.
  - The World Alliance of YMCAs is the highest governing body of the YMCA movement.
  - The World Alliance of YMCAs is headquartered in Geneva, Switzerland.
  - Recreation centers and gyms are common facilities found in YMCA locations.
  - YMCA locations often offer various programs and services to their members and the local community.

### 10. `48da75d87c66754ccc2e` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا پرل ہاربر پر حملے کے دوران امریکی ہاؤس اسپیکر کے طور پر خدمات انجام دینے والی واحد خاتون زندہ تھیں؟
- Gold evidence titles (R4_rows, normalised): attack on pearl harbor; nancy pelosi; speaker of the united states house of representatives
- R4 hits: speaker of the united states house of representatives
- Teacher entities: Pearl Harbor; Speaker of the United States House of Representatives
- Teacher facts (6):
  - The attack on Pearl Harbor occurred on December 7, 1941.
  - The Speaker of the United States House of Representatives is the presiding officer of the United States House of Representatives.
  - As of 1941, the Speaker of the United States House of Representatives was Sam Rayburn.
  - Sam Rayburn was a male politician.
  - The first female Speaker of the United States House of Representatives was Nancy Pelosi, who took office in 2007.
  - Nancy Pelosi was not a member of the U.S. House of Representatives in 1941.

### 11. `4de177ae8c827fd0ffb1` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا خوردہ ایسا کام ہے جس کے لیے کوئی بھی موزوں ہو سکتا ہے؟
- Gold evidence titles (R4_rows, normalised): disability; retail; retail clerk
- R4 hits: retail
- Teacher entities: Retail; Employment
- Teacher facts (6):
  - Retail is the sale of goods and services to consumers for personal or household use.
  - Retail involves various tasks such as managing inventory, handling customer transactions, and maintaining store displays.
  - Employment in retail often requires strong communication and interpersonal skills.
  - Many retail jobs involve working with customers, handling cash and operating a point-of-sale system.
  - Some retail positions may require specialized knowledge or skills, such as in electronics or clothing sales.
  - On-the-job training is common in the retail industry, and many employers provide training programs for new employees.

### 12. `5897ec22db850f7b416e` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا 2020 تک جان کی کے بعد زیادہ خواتین وزیر اعظم بنی ہیں بہ نسبت ان کے پہلے؟
- Gold evidence titles (R4_rows, normalised): helen clark; jacinda ardern; jenny shipley; john key; prime minister of new zealand
- R4 hits: jacinda ardern
- Teacher entities: List of female heads of government; Jacinda Ardern
- Teacher facts (6):
  - Jacinda Ardern is the Prime Minister of New Zealand.
  - Jacinda Ardern became the Prime Minister of New Zealand in 2017.
  - Jacinda Ardern is a female head of government.
  - The list of female heads of government includes all women who have held the top government office in their respective countries.
  - Jacinda Ardern was the third female Prime Minister of New Zealand.
  - The number of female heads of government has been increasing over the years.

### 13. `598be3dcddbd7775a827` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا فرضی طور پر باب مارلے کے بچے کبلائی خان کے بچوں کے خلاف رسی کشی جیت سکتے ہیں؟
- Gold evidence titles (R4_rows, normalised): bob marley; kublai khan; toghon (son of kublai)
- R4 hits: bob marley
- Teacher entities: Bob Marley; Genghis Khan; Tug of war
- Teacher facts (6):
  - Bob Marley was a Jamaican singer, songwriter, and musician.
  - Genghis Khan was the founder and emperor of the Mongol Empire.
  - Genghis Khan had many children, including at least six sons with his wife Borte.
  - Bob Marley had 11 acknowledged children with seven different women.
  - Tug of war is a sport that involves two teams pulling on opposite ends of a rope.
  - The objective of tug of war is to pull the rope a certain distance or make the opposing team lose their footing.

### 14. `660bfdc17b47f42facac` — gold **yes**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا مووس کو ارینڈیل کے سلطنت کے قریب کام کے لیے استعمال کیا جاتا ہے؟
- Gold evidence titles (R4_rows, normalised): arendelle: world of frozen; elsa (frozen); frozen (2013 film); moose
- R4 hits: (none)
- Teacher entities: Mouse; Arendelle
- Teacher facts (6):
  - Arendelle is a fictional kingdom and the main setting of Disney's 2013 animated film Frozen.
  - Mice are small rodents that are often used as working animals in various settings.
  - In the context of Frozen, Arendelle is a kingdom with a strong economy and infrastructure.
  - Mice are not typically associated with the Frozen franchise or the kingdom of Arendelle.
  - Arendelle is a fictional place and does not exist in real life.
  - Mice are commonly used in various industries such as research, entertainment, and companionship.

### 15. `7870b1cef39a4f685911` — gold **yes**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا آدم سینڈلر کو کول سپروس اور ایک سکوبا آدمک کے حوالے سے کوئی بات سمجھ آئے گی؟
- Gold evidence titles (R4_rows, normalised): big daddy (1999 film)
- R4 hits: (none)
- Teacher entities: Adam Sandler; Cold Sore; Scuba Diving
- Teacher facts (6):
  - Adam Sandler is an American actor, comedian, and film producer.
  - A cold sore is a viral infection caused by the herpes simplex virus.
  - Scuba diving is a mode of underwater diving that uses scuba equipment.
  - Adam Sandler has starred in numerous comedy films, often playing humorous roles.
  - Cold sores are typically characterized by the appearance of small, fluid-filled blisters.
  - Scuba diving requires specialized training and equipment to breathe underwater.

### 16. `80aa769f55b14c1e4d8d` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا سیبل وولورین سے متعلق ہیں؟
- Gold evidence titles (R4_rows, normalised): gulo; marten; sable; wolverine
- R4 hits: wolverine
- Teacher entities: Sibel; Wolverine
- Teacher facts (6):
  - Sibel is a Turkish given name.
  - Sibel is also the name of a genus of rodents.
  - Wolverine is a common name for the mammal Gulo gulo.
  - Wolverine is also the name of a fictional superhero in the Marvel Comics universe.
  - Sibel is not a known synonym for the wolverine animal.
  - The name Sibel is of Turkish origin, while the wolverine animal is native to the Northern Hemisphere.

### 17. `9576ab36999e196b87bb` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا پانی کے بغیر پورے ایسٹ اینڈرز سیریز کو بِنچ کرنا ممکن ہے؟
- Gold evidence titles (R4_rows, normalised): dehydration; eastenders; survival skills
- R4 hits: eastenders
- Teacher entities: EastEnders; Water
- Teacher facts (6):
  - EastEnders is a British soap opera that has been broadcast on BBC One since 1985.
  - EastEnders is set in the fictional London Borough of Walford in the East End of London.
  - The show is known for its portrayal of working-class life in London's East End.
  - Water is essential for human survival and is used in various ways in daily life, including drinking, cooking, and hygiene.
  - The UK has a well-established water supply system, with most households having access to clean drinking water.
  - Film and television production often requires significant resources, including water, to create realistic scenes and settings.

### 18. `987260ffef60cb2c5439` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا پرل ہاربر شارک دیوی کا خیالی گھر ہے؟
- Gold evidence titles (R4_rows, normalised): pearl harbor
- R4 hits: pearl harbor
- Teacher entities: Pearl Harbor; Shark Girl
- Teacher facts (6):
  - Pearl Harbor is a lagoon harbor on the island of Oahu, Hawaii, United States.
  - Pearl Harbor is the headquarters of the United States Pacific Fleet.
  - Shark Girl is a 2013 novel by Kelly Bingham.
  - Shark Girl is also the title of a 2004 novel by Betsy Petersen.
  - There are multiple works of fiction with the title Shark Girl.
  - Pearl Harbor is not commonly associated with sharks or shark fiction.

### 19. `a0896de3fd13cd0f3e16` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا اٹلانٹک کوڈ ویجی مائٹ سینڈوچ میں پایا جاتا ہے؟
- Gold evidence titles (R4_rows, normalised): animal; atlantic cod; fish; vegemite; yeast extract
- R4 hits: atlantic cod
- Teacher entities: Atlantic cod; Cod as food; Sandwich
- Teacher facts (6):
  - Atlantic cod is a mild-flavored fish often used in cooking.
  - Cod as food is a popular ingredient in many cuisines, particularly in Europe and North America.
  - A sandwich is a food item typically consisting of two or more slices of bread with fillings between them.
  - Cod is a common filling in fish sandwiches.
  - Atlantic cod is a species of fish found in the North Atlantic Ocean.
  - Fish sandwiches often feature battered and fried fish fillets.

### 20. `a54b16376f82a6e14bf9` — gold **no**, A_r2 no, TF yes, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا زمین کے ماحول میں پتھر تیرے گا؟
- Gold evidence titles (R4_rows, normalised): atmosphere; atmosphere of earth; granite; nitrogen; rock (geology); sandstone; troposphere
- R4 hits: (none)
- Teacher entities: Earth's atmosphere; Stone
- Teacher facts (6):
  - The Earth's atmosphere is a layer of gases surrounding the planet.
  - The Earth's atmosphere is held in place by the planet's gravity.
  - The density of the Earth's atmosphere varies with altitude.
  - A stone is a small rock or pebble.
  - Stones are subject to the force of gravity.
  - The terminal velocity of an object falling through the Earth's atmosphere depends on its mass and shape.

### 21. `b8df6ac181c4c5b280f9` — gold **no**, A_r2 no, TF —, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا بوتلنوز ڈولفن نظریاتی طور پر انسانی خواتین سے زیادہ نسل پیدا کر سکتے ہیں؟
- Gold evidence titles (R4_rows, normalised): bottlenose dolphin; gestation; human; pregnancy
- R4 hits: bottlenose dolphin; human
- Teacher entities: Bottlenose Dolphin; Human; Reproduction
- Teacher facts (6):
  - Bottlenose dolphins are known to form long-lasting social bonds with other dolphins.
  - Female bottlenose dolphins typically give birth to a single calf after a gestation period of almost 12 months.
  - Human females typically give birth to a single child after a gestation period of almost 9 months.
  - Bottlenose dolphins have been observed exhibiting complex behaviors and social structures.
  - The reproductive rate of bottlenose dolphins is influenced by various factors, including food availability and environmental conditions.
  - Humans have a relatively slow reproductive rate compared to some other mammals.

### 22. `d47120efd0f09badd848` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا مینٹھول کرسمس سے منسلک ہے؟
- Gold evidence titles (R4_rows, normalised): candy cane; menthol; peppermint
- R4 hits: menthol
- Teacher entities: Menthol; Christmas
- Teacher facts (6):
  - Menthol is a waxy, crystalline substance, clear or white in color, which is solid at room temperature.
  - Menthol is used in a variety of products, including candies, cough drops, and topical pain relief creams.
  - Christmas is an annual festival commemorating the birth of Jesus Christ, observed primarily on December 25 by Christians around the world.
  - Peppermint candies and other sweets are traditionally associated with Christmas in many countries.
  - Menthol is a common ingredient in peppermint candies and other sweets.
  - Peppermint flavoring is often used in holiday treats and drinks during the Christmas season.

### 23. `dabf94934721c26b2422` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا سیاہ دم والے خرگوش یورپی جنگلی بلی سے ڈرتے ہیں؟
- Gold evidence titles (R4_rows, normalised): black-tailed jackrabbit; european wildcat
- R4 hits: european wildcat
- Teacher entities: European Rabbit; European Wildcat
- Teacher facts (6):
  - The European Rabbit is a species of rabbit native to the Iberian Peninsula and western France.
  - European Rabbits are social animals and live in large groups in the wild.
  - The European Wildcat is a subspecies of wildcat that inhabits forests of Europe and the Caucasus.
  - European Wildcats are solitary and primarily nocturnal animals.
  - European Wildcats are carnivores and primarily feed on small mammals, including rabbits.
  - In the wild, European Rabbits have a number of natural predators, including wildcats, foxes, and birds of prey.

### 24. `e12fef0504a959e49b23` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا نیو انگلینڈ کے کسی شخص کو کافی اگا کر منافع ہو سکتا ہے؟
- Gold evidence titles (R4_rows, normalised): climate of new england; coffee; coffee bean; england
- R4 hits: coffee
- Teacher entities: New England; Coffee
- Teacher facts (6):
  - New England is a region in the northeastern United States.
  - New England's climate is humid continental, with cold winters and warm summers.
  - Coffee is a tropical crop that typically requires a subtropical or tropical climate to grow.
  - The United States is a significant consumer of coffee, but most of its coffee is imported.
  - Some coffee varieties can be grown in greenhouses or indoor containers in cooler climates.
  - New England has a number of greenhouses and indoor farms that grow a variety of crops.

### 25. `e6d3973ed3feb8a42928` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا افریقہائزڈ مکھیاں کثیر الثقافتی سمجھی جا سکتی ہیں؟
- Gold evidence titles (R4_rows, normalised): african bee; africanized bee; honey bee; multiculturalism; multiracial people; western honey bee
- R4 hits: africanized bee; multiculturalism
- Teacher entities: Africanized bee; Multiculturalism
- Teacher facts (6):
  - Africanized bees are a hybrid species of bee created by cross-breeding the European honey bee with the African honey bee.
  - Africanized bees are known for their aggressive behavior when threatened or provoked.
  - Multiculturalism is the coexistence of diverse cultures within a single society or community.
  - Multiculturalism can be understood in different ways, including as a demographic fact, a cultural policy, or an ideology.
  - Africanized bees have been introduced to various parts of the world, including the Americas, and have established feral colonies.
  - The behavior and social structure of Africanized bees are influenced by their genetic makeup and environmental factors.

### 26. `f43533225534420816d6` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا آپ سولجر فیلڈ میں دیوقامت اسکویڈ رکھ سکتے ہیں؟
- Gold evidence titles (R4_rows, normalised): american football; giant squid; gridiron football; soldier field
- R4 hits: soldier field
- Teacher entities: Soldier Field; Squid
- Teacher facts (6):
  - Soldier Field is a stadium located in Chicago, Illinois, United States.
  - Soldier Field is situated on Lake Shore Drive, adjacent to Lake Michigan.
  - Squid are marine cephalopods with a soft, boneless body.
  - Squid are generally found in the ocean, typically at depths of 200-700 meters.
  - Squid are not typically found in freshwater environments, such as the Great Lakes.
  - Soldier Field has hosted various events, including sports games and concerts.

### 27. `f8e6087cd52c925fedad` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا قدیم ٹونی بینیٹ 2020 میں بچہ پیدا کر سکتے تھے؟
- Gold evidence titles (R4_rows, normalised): ramjit raghav; tony bennett
- R4 hits: tony bennett
- Teacher entities: Tony Bennett; 2020
- Teacher facts (6):
  - Tony Bennett is an American singer of popular music, standards, and jazz.
  - Tony Bennett was born on August 3, 1926.
  - Tony Bennett is still active in his music career despite his advanced age.
  - The average age of fathers at the birth of their child varies significantly across the world.
  - There is no maximum age limit for men to father a child, but fertility declines with age.
  - Tony Bennett's wife, Susan Crow, is 54 years younger than him.

### 28. `ff3811735ededd8ec3a7` — gold **no**, A_r2 no, TF yes, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا سلائم للیز اسپیرگس سے مختلف سائنسی خاندان میں ہیں؟
- Gold evidence titles (R4_rows, normalised): albuca; asparagaceae; asparagus
- R4 hits: (none)
- Teacher entities: Salim Ali; Lalita Kumari; Sylvia Sparrow; Family (biology)
- Teacher facts (6):
  - Salim Ali was an Indian ornithologist and naturalist.
  - Lalita Kumari was an Indian ornithologist and wife of Salim Ali.
  - Sylvia Sparrow is a fictional character in the animated TV series 'The Sylvester and Tweety Mysteries'.
  - Salim Ali and Lalita Kumari were married in 1939.
  - Salim Ali was a key figure in Indian ornithology and conservation.
  - In biology, a family is a level of classification that ranks between order and genus.

## B. The 38 TF wins (TF right, A_r2 wrong)

By (gold, A_r2 pred): gold no → A_r2 yes: 20; gold yes → A_r2 no: 13; gold no → A_r2 unparsed: 3; gold yes → A_r2 unparsed: 2. R4 hit yes: 31, no: 7.

### 1. `05e9cdc44f1b066badd7` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا خواتین میں مونچھیں ہونا عام بات ہے؟
- Gold evidence titles (R4_rows, normalised): beard; facial hair; moustache
- R4 hits: moustache
- Teacher entities: Women; Moustache
- Teacher facts (6):
  - Women is the plural form of woman, an adult human female.
  - A moustache is a growth of hair on the upper lip.
  - Hirsutism is a condition characterized by excessive growth of terminal hair in certain areas of the body, including the face, in women.
  - Hypertrichosis is a condition characterized by excessive growth of terminal hair in certain areas of the body, including the face, in both men and women.
  - Women with hirsutism may develop a moustache or other facial hair.
  - The prevalence of hirsutism in women varies depending on the population being studied and the criteria used to define the condition.

### 2. `0fe3edf35bd1a1e662a9` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا گینڈالف فرضی طور پر چارمد کے بارباس کے لیے ایک زبردست دشمن ہے؟
- Gold evidence titles (R4_rows, normalised): barbas (charmed); gandalf; marbas
- R4 hits: gandalf
- Teacher entities: Gandalf; Barbossa; Pirates of the Caribbean
- Teacher facts (6):
  - Gandalf is a fictional character in J. R. R. Tolkien's Middle-earth legendarium.
  - Gandalf is a powerful wizard who plays a key role in The Hobbit and The Lord of the Rings.
  - Barbossa is a fictional character in the Pirates of the Caribbean film series.
  - Barbossa is the main antagonist in the first film, The Curse of the Black Pearl.
  - Gandalf and Barbossa are characters from different fictional universes.
  - There is no direct connection between Gandalf and Barbossa in their respective fictional stories.

### 3. `30fcf62f58849cb4ab05` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا عظیم اہرام ایک خلافتی حکومت نے تعمیر کیے تھے؟
- Gold evidence titles (R4_rows, normalised): ancient egypt; giza pyramid complex; great pyramid of giza; theocracy
- R4 hits: great pyramid of giza; theocracy
- Teacher entities: Great Pyramid of Giza; Theocracy
- Teacher facts (6):
  - The Great Pyramid of Giza is located in modern-day Egypt and is believed to have been built as a tomb for Pharaoh Khufu.
  - The Great Pyramid of Giza is estimated to have been constructed over a period of 20 years, with a workforce of around 100,000 laborers.
  - The ancient Egyptian government was a complex system that included both secular and religious elements.
  - The pharaoh was considered to be a divine ruler, with a direct connection to the gods.
  - The Great Pyramid of Giza was built during the Old Kingdom period of Ancient Egypt, specifically during the reign of the 4th dynasty.
  - The concept of theocracy typically refers to a system of government in which the laws and policies are based on religious doctrine.

### 4. `3486b6b5c69878bb088d` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ایریزونا میں کیلیفورنیا کے مقابلے میں زیادہ گوبھی اگتی ہے؟
- Gold evidence titles (R4_rows, normalised): arizona; california; cauliflower; climate of the united states
- R4 hits: arizona; california
- Teacher entities: Arizona; California; Cabbage
- Teacher facts (6):
  - Arizona is a state in the southwestern United States.
  - California is a state on the West Coast of the United States.
  - Cabbage is a cool-season crop that can be grown in many parts of the United States.
  - Arizona's climate is generally warmer and drier than California's.
  - The United States Department of Agriculture (USDA) tracks data on cabbage production by state.
  - According to the USDA, California is consistently one of the top cabbage-producing states in the country.

### 5. `3503e8d41364eb0b33bc` — gold **no**, A_r2 —, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا ایک ذمہ دار بارٹینڈر میلی بوبی براؤن کے لیے مشروب بنائے گا؟
- Gold evidence titles (R4_rows, normalised): legal drinking age; millie bobby brown; national minimum drinking age act
- R4 hits: (none)
- Teacher entities: Milly Bobby Brown; Bartender
- Teacher facts (6):
  - Milly Bobby Brown is a British actress and model.
  - Milly Bobby Brown was born on February 19, 2004.
  - A bartender is a person who formulates and serves alcoholic drinks behind the bar in a licensed establishment.
  - Bartenders are responsible for checking the age of customers before serving them alcohol.
  - In many countries, there are laws regulating the minimum age at which a person can be served alcohol.
  - The minimum drinking age varies from country to country.

### 6. `3b38e00d8d1da26f7f02` — gold **no**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا ماریو تیز دوڑنے کے لیے مشروم استعمال کرتا ہے؟
- Gold evidence titles (R4_rows, normalised): goomba; super mario
- R4 hits: (none)
- Teacher entities: Mario; Super Mushroom
- Teacher facts (6):
  - Mario is a fictional character in the Super Mario video game series.
  - In the Super Mario series, Mario is known for his ability to run and jump.
  - The Super Mushroom is a power-up item in the Super Mario series.
  - When Mario consumes a Super Mushroom, he grows bigger and gains additional abilities.
  - The Super Mushroom does not directly affect Mario's running speed.
  - Power-ups in the Super Mario series are items that give Mario new abilities or enhance his existing ones.

### 7. `3d7050c0e9bfbeef27c1` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا میلانیا ٹرمپ کا پیشہ اولگا کیوریلنکو کے پیشے کے برابر تھا؟
- Gold evidence titles (R4_rows, normalised): melania trump; olga kurylenko; olga storozhenko
- R4 hits: melania trump; olga kurylenko
- Teacher entities: Melania Trump; Olga Kurylenko
- Teacher facts (6):
  - Melania Trump is a former model and the wife of the 45th U.S. President Donald Trump.
  - Melania Trump began her modeling career at a young age and worked with several top designers and photographers.
  - Olga Kurylenko is a French actress and model of Ukrainian descent.
  - Olga Kurylenko has appeared in numerous films, including the James Bond movie Quantum of Solace.
  - Before becoming an actress, Olga Kurylenko worked as a model, posing for several prominent fashion designers and magazines.
  - Both Melania Trump and Olga Kurylenko have worked in the fashion industry as models.

### 8. `4760fe3d0c80beb30edd` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): اگر کوئی شخص ویگن ہو تو کیا وہ شہد کھائے گا؟
- Gold evidence titles (R4_rows, normalised): honey; veganism
- R4 hits: honey; veganism
- Teacher entities: Veganism; Honey
- Teacher facts (6):
  - Veganism is a lifestyle and dietary choice that seeks to exclude, as far as possible, all forms of exploitation of animals for food, clothing, and other purposes.
  - Vegans do not consume or use products that come from animals, including eggs, dairy products, and even honey.
  - Honey is an animal byproduct produced by bees.
  - Some vegans choose to avoid honey because it is an animal product.
  - Vegan diets and lifestyles vary widely among individuals, and not all vegans agree on what products are acceptable to consume or use.
  - The decision to consume honey is often a personal choice for individuals who follow a vegan lifestyle.

### 9. `4810297f474b0e97ed78` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا دس گیلن سمندری پانی چھ سالہ بچے کو دبا سکتا ہے؟
- Gold evidence titles (R4_rows, normalised): child; early childhood; seawater; weigh station
- R4 hits: child; seawater
- Teacher entities: Gallon; Seawater; Child
- Teacher facts (6):
  - A gallon is a unit of volume in both the US customary and imperial systems.
  - Seawater is a complex, variable mixture of salts, water, and other substances.
  - The density of seawater is approximately 1.03-1.04 g/ml, depending on temperature and salinity.
  - The average weight of a six-year-old child varies by sex and other factors, but is generally around 45-60 pounds (20-27 kg).
  - The volume of a person's lungs when fully inflated is approximately 4-6 liters.
  - The pressure exerted by a column of fluid is proportional to the density of the fluid, the acceleration due to gravity, and the height of the column.

### 10. `4ec8cd118be8a53ec516` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا الیگزینڈر ہیملٹن کو کوالاز کے بارے میں معلوم ہوتا؟
- Gold evidence titles (R4_rows, normalised): alexander hamilton; koala
- R4 hits: alexander hamilton
- Teacher entities: Alexander Hamilton; Cowpox
- Teacher facts (6):
  - Alexander Hamilton was a founding father of the United States.
  - Cowpox is a viral disease affecting cattle.
  - Edward Jenner developed the first vaccine against smallpox using cowpox in 1796.
  - Alexander Hamilton died in 1804.
  - Cowpox is closely related to the virus that causes smallpox.
  - Vaccination against smallpox was not widely practiced in the United States during Alexander Hamilton's lifetime.

### 11. `502d4dae6f08e73a5569` — gold **yes**, A_r2 no, TF yes, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا ہوم ڈیپو سلیب پر چڑھانے کے سامان کے لیے ایک جامع دکان ہے؟
- Gold evidence titles (R4_rows, normalised): crucifixion; descriptions in antiquity of the execution cross; one stop shop; the home depot
- R4 hits: (none)
- Teacher entities: Home Depot; Sleek
- Teacher facts (6):
  - Home Depot is an American multinational home improvement retail corporation.
  - Home Depot operates many big-box stores across the United States, all 10 provinces of Canada, and the country of Mexico.
  - Home Depot sells building materials, home improvement products, lawn and garden supplies, and tools.
  - Sleek is a brand of cookware and kitchen appliances.
  - Sleek products are available at various retailers, both online and in-store.
  - Home Depot carries a variety of cookware and kitchen appliances from different brands.

### 12. `5e2d9519c7c521411817` — gold **yes**, A_r2 —, TF yes, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا ٹی ایم این ٹی رنگ بھرنے کی کتاب میں پیزا ہوگا؟
- Gold evidence titles (R4_rows, normalised): teenage mutant ninja turtles
- R4 hits: (none)
- Teacher entities: TMNT; Coloring book; Pizza
- Teacher facts (6):
  - Teenage Mutant Ninja Turtles (TMNT) is a popular American media franchise created by Kevin Eastman and Peter Laird.
  - Coloring books are widely used for entertainment and educational purposes, often featuring popular characters.
  - Pizza is a popular Italian dish made from flatbread, typically topped with ingredients like cheese, tomato sauce, and various meats or vegetables.
  - TMNT characters, including Leonardo, Michelangelo, Donatello, and Raphael, are known for their love of pizza.
  - TMNT has been featured in various forms of media, including comic books, animated television series, and films.
  - Coloring books often include images and characters from popular culture, such as cartoons, movies, and television shows.

### 13. `690a36d5fed5b17969aa` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا جیک ڈیمپسی نے موجودہ WBC ہیوی ویٹ چیمپئن سے مقابلہ کیا تھا؟
- Gold evidence titles (R4_rows, normalised): deontay wilder; jack dempsey; tyson fury
- R4 hits: jack dempsey
- Teacher entities: Jack Dempsey; World Boxing Council; Heavyweight
- Teacher facts (6):
  - Jack Dempsey was an American professional boxer who competed from 1914 to 1927.
  - The World Boxing Council (WBC) is one of four major organizations which sanction professional boxing bouts.
  - The WBC Heavyweight Championship is a professional boxing championship contested between top-ranked boxers in the heavyweight division.
  - Jack Dempsey held the World Heavyweight Championship title from 1919 to 1926.
  - The current WBC Heavyweight Champion is determined by the outcome of bouts sanctioned by the WBC.
  - Jack Dempsey's boxing career ended in 1927.

### 14. `70b2122323424f6e80c2` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا لیبرون جیمز فرضی طور پر یوری گاگرین کی طرف اوپر دیکھے گا؟
- Gold evidence titles (R4_rows, normalised): lebron james; yuri gagarin
- R4 hits: lebron james; yuri gagarin
- Teacher entities: LeBron James; Yuri Gagarin
- Teacher facts (6):
  - LeBron James is an American professional basketball player.
  - Yuri Gagarin was a Soviet pilot and cosmonaut who became the first human to journey into outer space.
  - Yuri Gagarin orbited the Earth once on April 12, 1961.
  - LeBron James is 6 feet 8 inches (2.06 m) tall.
  - Yuri Gagarin was 5 feet 7 inches (1.70 m) tall.
  - Yuri Gagarin died on March 27, 1968, in a plane crash.

### 15. `71ca1a8a75c5c582ab3f` — gold **no**, A_r2 —, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا زیادہ تر ہائی اسکول کے ہیڈ کوچز این سی ایس یو کے ہیڈ کوچ جتنا کماتے ہیں؟
- Gold evidence titles (R4_rows, normalised): head coach; nc state wolfpack; nc state wolfpack football
- R4 hits: head coach
- Teacher entities: High school; Head coach; NCAA
- Teacher facts (6):
  - A high school is an educational institution where students typically complete secondary education.
  - A head coach is the highest coaching position within a school's athletic department.
  - The National Collegiate Athletic Association (NCAA) is a non-profit organization that regulates student athletics among its member colleges in the United States.
  - NCAA head coaches are responsible for leading their teams in their respective sports.
  - High school head coaches are typically employed by their school districts or schools.
  - Salaries for high school head coaches and NCAA head coaches vary widely depending on factors such as location, sport, and experience.

### 16. `73c52134ab2a903d86db` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا لیموں دودھ کے ذائقے کو بہتر بناتا ہے؟
- Gold evidence titles (R4_rows, normalised): curdling; lemon; soured milk
- R4 hits: lemon
- Teacher entities: Lemon; Milk
- Teacher facts (6):
  - Lemons are a good source of citric acid and ascorbic acid.
  - Citric acid is a natural preservative and is often used to add an acidic (sour) taste to foods and drinks.
  - Milk is a nutrient-rich liquid produced by mammary glands of female mammals.
  - The taste of milk can be affected by the presence of acidic or alkaline substances.
  - Adding lemon juice to milk can cause it to curdle or separate.
  - Curdling can affect the texture and appearance of milk, but not necessarily its nutritional value.

### 17. `7419826e4373eb1e83ff` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا کتا گرے سیل سے پہلے گھنٹی کی آواز پر ردعمل ظاہر کرے گا؟
- Gold evidence titles (R4_rows, normalised): dog; dog anatomy; grey seal; hearing range; hertz; pinniped
- R4 hits: dog; grey seal
- Teacher entities: Dog; Grey seal; Bell
- Teacher facts (6):
  - Dogs are known for their ability to hear sounds at higher frequencies than humans.
  - Grey seals are found in the North Atlantic and are known for their distinctive barks and grunts.
  - Bells produce a ringing sound when struck, which can be heard by both humans and animals.
  - Dogs have been observed responding to various sounds, including bells, in different contexts.
  - The hearing range of dogs is approximately 40 Hz to 40,000 Hz.
  - Grey seals do not typically make sounds that resemble the ringing of a bell.

### 18. `7571100f05bc56919c78` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا جنازے میں ہیپی ہارڈکور موسیقی بجانا غیر معمولی ہے؟
- Gold evidence titles (R4_rows, normalised): dirge; funeral; funeral march; happy hardcore
- R4 hits: funeral; happy hardcore
- Teacher entities: Funeral; Happy hardcore
- Teacher facts (6):
  - A funeral is a ceremony for celebrating the life of a deceased person.
  - Funerals often include music, but the type of music varies across cultures and traditions.
  - Happy hardcore is a subgenre of hardcore techno characterized by fast-paced rhythms and often cheerful melodies.
  - Happy hardcore originated in the 1990s in Europe, particularly in the UK and the Netherlands.
  - Funeral music is typically somber and mournful, but some cultures incorporate more upbeat music to celebrate the deceased's life.
  - There is no universal standard for funeral music, and choices often depend on the deceased's preferences, cultural background, and personal relationships.

### 19. `7b84d2bc643ddc2085f0` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا نوح کی کشتی مفروضاتی طور پر سیلاب زدہ لنکن سرنگ سے گزر سکتی ہے؟
- Gold evidence titles (R4_rows, normalised): cubit; lincoln tunnel; noah's ark
- R4 hits: lincoln tunnel; noah's ark
- Teacher entities: Noah's Ark; Lincoln Tunnel
- Teacher facts (6):
  - Noah's Ark is a biblical vessel said to have been built by Noah to save himself, his family, and two of every kind of animal from a worldwide flood.
  - The size of Noah's Ark is described in the Bible as being 300 cubits in length, 50 cubits in width, and 30 cubits in height.
  - The Lincoln Tunnel is a 1.5-mile-long (2.4 km) vehicular tunnel under the Hudson River, connecting Weehawken, New Jersey, to Midtown Manhattan in New York City.
  - The tunnel's maximum height clearance is 13 feet (4 meters) and its maximum width is 21.5 feet (6.6 meters).
  - The tunnel's roadways are 21.5 feet (6.6 meters) wide, with two lanes in each direction.
  - The tunnel's maximum vehicle height and width restrictions are strictly enforced due to its limited clearance and narrow roadways.

### 20. `7c3759cc1da78e9fbd79` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا دیوالی کی تقریب میں کرسٹیشین شامل ہو سکتا ہے؟
- Gold evidence titles (R4_rows, normalised): buddhist cuisine; diet in hinduism; diwali; hinduism; lobster
- R4 hits: diwali
- Teacher entities: Diwali; Christianity
- Teacher facts (6):
  - Diwali is a significant festival in Hinduism, celebrated over five days.
  - Diwali is also celebrated by Sikhs and Jains.
  - Christianity is a monotheistic Abrahamic religion based on the life and teachings of Jesus Christ.
  - Diwali is not a part of Christian religious traditions.
  - Many Christians in India participate in Diwali celebrations as a cultural event.
  - Diwali is a national holiday in several countries, including India, Nepal, and Sri Lanka.

### 21. `867a5425c26092b30fbf` — gold **no**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا ڈزنی کے سات بونے اصل ہیں؟
- Gold evidence titles (R4_rows, normalised): seven dwarfs; snow white and the seven dwarfs (1937 film)
- R4 hits: (none)
- Teacher entities: The Walt Disney Company; Snow White and the Seven Dwarfs
- Teacher facts (6):
  - The Walt Disney Company is an American multinational mass media and entertainment conglomerate.
  - Snow White and the Seven Dwarfs is a 1937 American animated musical fantasy film produced by Walt Disney Productions.
  - The Seven Dwarfs are the main characters in the film Snow White and the Seven Dwarfs.
  - The names of the Seven Dwarfs are Doc, Grumpy, Happy, Sleepy, Bashful, Sneezy, and Dopey.
  - The Seven Dwarfs were created by Walt Disney and animator Albert Hurter.
  - The film Snow White and the Seven Dwarfs is based on the German fairy tale Snow White by the Brothers Grimm.

### 22. `8d708927b1593d8b9b42` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ووڈرو ولسن پلسسی بمقابلہ فرگوسن کے فیصلے کی حمایت کریں گے؟
- Gold evidence titles (R4_rows, normalised): plessy v. ferguson; woodrow wilson
- R4 hits: plessy v. ferguson; woodrow wilson
- Teacher entities: Woodrow Wilson; Plessy v. Ferguson
- Teacher facts (6):
  - Woodrow Wilson was the 28th President of the United States, serving from 1913 to 1921.
  - Plessy v. Ferguson was a landmark U.S. Supreme Court decision made in 1896.
  - The Plessy v. Ferguson decision upheld the constitutionality of racial segregation laws for public facilities as long as the facilities were equal in quality.
  - Woodrow Wilson was a Democrat and the leader of the Progressive Movement in the early 20th century.
  - During Wilson's presidency, he supported several progressive reforms, including the creation of the Federal Reserve System and the passage of the 19th Amendment.
  - Despite his progressive views, Wilson has been criticized for his handling of racial issues, including his support for racial segregation in federal workplaces.

### 23. `9b491a31c4212e70b18c` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا جین آسٹن کو وسطی بچے کے سنڈروم کا سامنا تھا؟
- Gold evidence titles (R4_rows, normalised): jane austen; middle child syndrome; timeline of jane austen
- R4 hits: jane austen; middle child syndrome
- Teacher entities: Jane Austen; Middle child syndrome
- Teacher facts (6):
  - Jane Austen was an English novelist known for her works of romantic fiction.
  - Jane Austen was born on December 16, 1775, in Steventon, Hampshire, England.
  - Jane Austen was the seventh of eight children born to Reverend George Austen and Cassandra Austen.
  - Middle child syndrome is a term used to describe the perceived effects of being a middle child on a person's personality and behavior.
  - The concept of middle child syndrome is not universally accepted by psychologists and researchers.
  - Jane Austen's life and works have been extensively studied and documented by scholars and biographers.

### 24. `9e78c01675a198781cca` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا مائیکل بلومبرگ مائکرونیشیا کا قرضہ ایک دہائی کے لیے ادا کر سکتے ہیں؟
- Gold evidence titles (R4_rows, normalised): economy of the federated states of micronesia; michael bloomberg; micronesia
- R4 hits: michael bloomberg
- Teacher entities: Michael Bloomberg; Federated States of Micronesia; Decade
- Teacher facts (6):
  - Michael Bloomberg is an American businessman, politician, and philanthropist.
  - As of 2022, Michael Bloomberg's net worth is estimated to be over $76 billion.
  - The Federated States of Micronesia is a country in the Pacific Ocean with a population of around 113,000 people.
  - The Federated States of Micronesia has a GDP of approximately $390 million.
  - The country's debt-to-GDP ratio is around 30%.
  - A decade is a period of 10 years.

### 25. `a6f593e93ca87c53a0a1` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا سلیب پر چڑھانا امریکی آئین کی آٹھویں ترمیم کی خلاف ورزی ہے؟
- Gold evidence titles (R4_rows, normalised): crucifixion; cruel and unusual punishment; eighth amendment to the united states constitution; united states constitutional sentencing law
- R4 hits: crucifixion; eighth amendment to the united states constitution
- Teacher entities: Crucifixion; Eighth Amendment to the United States Constitution
- Teacher facts (6):
  - Crucifixion is a method of punishment in which the victim is tied or nailed to a wooden cross and left to hang, often until death.
  - The Eighth Amendment to the United States Constitution prohibits the federal government from imposing excessive bail, excessive fines, or cruel and unusual punishment.
  - The Eighth Amendment applies to the federal government, but the Due Process Clause of the Fourteenth Amendment has been used to apply the Eighth Amendment to state and local governments as well.
  - The Supreme Court has interpreted the Eighth Amendment to prohibit punishments that are inhumane, barbarous, or excessively severe.
  - The Supreme Court has also held that the Eighth Amendment prohibits punishments that are arbitrary, capricious, or not related to the severity of the crime.
  - The Supreme Court has the final say on the interpretation of the Eighth Amendment and what constitutes cruel and unusual punishment.

### 26. `acb7edac010554509519` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا انسانی نسل کلوروفل کے بغیر ختم ہو جائے گی؟
- Gold evidence titles (R4_rows, normalised): breathing; chlorophyll; photosynthesis
- R4 hits: chlorophyll
- Teacher entities: Human; Chlorophyll
- Teacher facts (6):
  - Humans are a species of primates that belong to the family Hominidae.
  - Chlorophyll is a green pigment found in plants, algae, and cyanobacteria that plays a crucial role in photosynthesis.
  - Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water.
  - Humans and other animals rely on plants and other organisms for food and oxygen.
  - Chlorophyll is essential for photosynthesis to occur in plants and other organisms.
  - The food chain is a series of events where one organism is eaten by another, and plants are the primary producers of the food chain.

### 27. `b77170e15c0b1bd72340` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا دورین کے پھل خوشبو کے لحاظ سے خوشگوار ہیں؟
- Gold evidence titles (R4_rows, normalised): durian
- R4 hits: durian
- Teacher entities: Durian; Fruit
- Teacher facts (6):
  - Durian is a tropical fruit native to Southeast Asia.
  - Durian is known for its distinctive odor and spiky exterior.
  - The smell of durian is often compared to rotten eggs, gasoline, or sewage.
  - Some people enjoy the smell and taste of durian, while others find it repulsive.
  - Durian is banned in many public places in Southeast Asia due to its strong odor.
  - The smell of durian is caused by the high concentration of trimethylamine in the fruit.

### 28. `c67945e7c471a9692c88` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا برفانی الو سونوران میں زندہ رہ سکتا ہے؟
- Gold evidence titles (R4_rows, normalised): arctic; climate of the arctic; snowy owl; sonoran desert
- R4 hits: snowy owl; sonoran desert
- Teacher entities: Snowy Owl; Sonoran Desert
- Teacher facts (6):
  - The Snowy Owl is a large, white owl native to the Arctic regions.
  - Snowy Owls primarily inhabit areas with low vegetation and few trees.
  - The Sonoran Desert is a desert region located in southwestern Arizona, southwestern New Mexico, and northern Mexico.
  - The Sonoran Desert is known for its hot and dry climate, with very little rainfall throughout the year.
  - Snowy Owls are adapted to cold, Arctic environments and have thick feathers to keep warm.
  - The average temperature in the Sonoran Desert can reach up to 120°F (49°C) during the summer months.

### 29. `d0145341694e7034555d` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا اقوام متحدہ کا موسمیاتی تبدیلی پر فریم ورک کنونشن جاویٹس سینٹر میں منعقد کیا جا سکتا ہے؟
- Gold evidence titles (R4_rows, normalised): javits center; united nations framework convention on climate change
- R4 hits: javits center; united nations framework convention on climate change
- Teacher entities: United Nations Framework Convention on Climate Change; Javits Center
- Teacher facts (6):
  - The United Nations Framework Convention on Climate Change is an international environmental treaty.
  - The Javits Center is a large convention center located in New York City.
  - The Javits Center has hosted numerous high-profile events, including conferences and exhibitions.
  - The United Nations Framework Convention on Climate Change has been ratified by almost all countries in the world.
  - The Javits Center has undergone several renovations and expansions since its opening in 1986.
  - The United Nations often holds conferences and meetings at various locations around the world.

### 30. `d15a33b8f88dc5ce32b0` — gold **no**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا ڈھولک نوازوں کو اضافی تاروں کی ضرورت ہوتی ہے؟
- Gold evidence titles (R4_rows, normalised): drum; drummer; string instrument
- R4 hits: (none)
- Teacher entities: Dholak; Stringed instrument
- Teacher facts (6):
  - A dholak is a type of hand drum used in Indian folk music.
  - Dholak is a rope-tuned, two-headed drum with a cylindrical shell.
  - The dholak is typically played with the fingers and palms of the hand.
  - Stringed instruments are musical instruments that produce sound when a string is plucked or strummed.
  - Some stringed instruments have a fixed number of strings, while others can have additional strings added.
  - The dholak is not typically classified as a stringed instrument.

### 31. `d1dbb41001ab4ad3b924` — gold **no**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا نیلے ہونٹ معمول کی بات ہے؟
- Gold evidence titles (R4_rows, normalised): cyanosis; oxygen saturation (medicine)
- R4 hits: (none)
- Teacher entities: Blue lips; Normality
- Teacher facts (6):
  - Blue lips, also known as cyanosis, is a condition characterized by a bluish discoloration of the skin and mucous membranes.
  - Cyanosis is often a sign of inadequate oxygenation of the blood.
  - Normality refers to the state of being within the range of typical or expected values for a particular characteristic or condition.
  - Cyanosis can be caused by a variety of factors, including respiratory and cardiovascular diseases.
  - In some cases, cyanosis can be a normal adaptation to high-altitude environments or intense exercise.
  - Cyanosis can also be a symptom of certain genetic disorders, such as methemoglobinemia.

### 32. `f087d393e2ab1f8783f5` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا 1980 کے امریکہ کے صدارتی انتخابات کے نتائج کو فوٹو فِنِش کہا جا سکتا ہے؟
- Gold evidence titles (R4_rows, normalised): 1980 united states presidential election; history of the united states (1980–1991)
- R4 hits: 1980 united states presidential election
- Teacher entities: 1980 United States presidential election; Photo finish
- Teacher facts (6):
  - The 1980 United States presidential election was held on November 4, 1980.
  - The election was between incumbent President Jimmy Carter and challenger Ronald Reagan.
  - Ronald Reagan won the election with 489 electoral votes to Jimmy Carter's 49.
  - A photo finish is a term used to describe a very close finish in a competition, often requiring a photograph to determine the winner.
  - In the context of elections, a photo finish would imply a very small margin of victory in the popular vote or a close contest in the electoral college.
  - The 1980 United States presidential election was not particularly close in the popular vote, with Ronald Reagan winning by nearly 10 percentage points.

### 33. `f2859b2ce17b5f5a6ad9` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا چنچیلا نسلِ فیلس کاتس چوہے کی قسم ہے؟
- Gold evidence titles (R4_rows, normalised): cat; chinchilla; persian cat; rodent
- R4 hits: chinchilla; rodent
- Teacher entities: Chinchilla; Rodent; Felis catus
- Teacher facts (6):
  - Chinchillas are small, fluffy rodents native to the Andes mountains in South America.
  - Chinchillas are members of the family Chinchillidae.
  - Felis catus is the domestic cat, a small, typically furry, carnivorous mammal.
  - Domestic cats are not rodents, but rather members of the family Felidae.
  - Rodents are a group of mammals characterized by a single pair of continuously growing incisors in each jaw.
  - Chinchillas are not members of the Felidae family.

### 34. `f4a336168f5165de7f0d` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا جولائی میں ہنگامی سرجری کی ضرورت رکھنے والا شمالی فر سیل محفوظ بے ہوشی کا امیدوار ہو سکتا ہے؟
- Gold evidence titles (R4_rows, normalised): anesthesia; northern fur seal
- R4 hits: northern fur seal
- Teacher entities: July; Emergency surgery; Northern fur seal; General anaesthesia
- Teacher facts (6):
  - July is a month in the Northern Hemisphere summer.
  - Emergency surgery is a surgical operation that is performed immediately to save a life or to prevent severe harm.
  - The northern fur seal is a species of pinniped that is found in the North Pacific Ocean.
  - General anaesthesia is a medical treatment that induces a reversible coma-like state, making patients unresponsive to pain and unable to recall the procedure.
  - Northern fur seals are susceptible to various health issues, including respiratory problems and injuries.
  - General anaesthesia can be used in veterinary medicine to facilitate surgical procedures in animals, including marine mammals.

### 35. `f7a32d14efd589015fe1` — gold **no**, A_r2 —, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ہرپیس سمپلیکس وائرس زہرہ پر پھیل سکتا ہے؟
- Gold evidence titles (R4_rows, normalised): herpes simplex virus; sterilization (microbiology); venus; virus
- R4 hits: herpes simplex virus; venus
- Teacher entities: Herpes simplex virus; Venus
- Teacher facts (6):
  - Herpes simplex virus is a human pathogen that primarily infects humans.
  - Herpes simplex virus is a member of the Herpesviridae family.
  - Venus is the second planet from the Sun in the Solar System.
  - Venus has a very hostile surface environment, with surface temperatures reaching up to 462°C.
  - There are no known human settlements or biological organisms on Venus.
  - Herpesviruses are generally specific to their host species and are not easily transmitted to other species.

### 36. `fba624d8c01833419760` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا اگست دنیا کے کسی حصے کے لیے سردیوں کا مہینہ ہے؟
- Gold evidence titles (R4_rows, normalised): august; summer; winter
- R4 hits: august
- Teacher entities: August; Season; Southern Hemisphere; Northern Hemisphere
- Teacher facts (6):
  - August is the eighth month of the year in the Julian and Gregorian calendars.
  - Seasons are caused by the tilt of the Earth's axis relative to its orbital plane around the Sun.
  - The Southern Hemisphere experiences winter while the Northern Hemisphere experiences summer during the months of June, July, and August.
  - The Northern Hemisphere experiences winter while the Southern Hemisphere experiences summer during the months of December, January, and February.
  - Some parts of the world, such as the Southern United States, have a mild winter season.
  - The Southern Hemisphere includes countries such as Australia, New Zealand, and South Africa.

### 37. `fc9339f21f44a3841a7c` — gold **yes**, A_r2 —, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا کوالا گوشت کی بجائے یوکلپٹس کو ترجیح دیتے ہیں؟
- Gold evidence titles (R4_rows, normalised): eucalypt; koala
- R4 hits: koala
- Teacher entities: Koala; Eucalyptus
- Teacher facts (6):
  - Koalas are marsupials native to Australia.
  - Koalas are herbivores, primarily feeding on leaves, shoots, and fruits.
  - Eucalyptus is a genus of flowering plants native to Australia and surrounding islands.
  - Eucalyptus leaves are toxic to many animals due to the presence of toxic compounds like phenolics and terpenes.
  - Koalas have a specialized digestive system that allows them to break down cellulose in plant material and detoxify some of the toxic compounds found in eucalyptus leaves.
  - Koalas rely heavily on eucalyptus leaves as their primary food source, with some species of eucalyptus being preferred over others.

### 38. `fdeec181aeee06303113` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا آپ کو کراچی میں صلیب نما نشان ملنے کا امکان ہے؟
- Gold evidence titles (R4_rows, normalised): christian cross; crucifix; karachi; muslims; religion in karachi
- R4 hits: karachi
- Teacher entities: Karachi; Cross
- Teacher facts (6):
  - Karachi is the largest city in Pakistan.
  - Karachi is a major port city in Sindh, Pakistan.
  - The Cross is a symbol with a long history in various cultures and religions.
  - The Cross is a common symbol in Christianity.
  - Karachi has a significant Christian population.
  - Karachi has several churches and Christian institutions.
