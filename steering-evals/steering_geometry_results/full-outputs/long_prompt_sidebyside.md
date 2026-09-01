# Longer-prompts × theme steering — side by side with the unsteered baseline

Qwen/Qwen2-1.5B · seed 0 · ntok 120 · steer recipe: `CONTRAST_MODE=logit ALPHA=2 DL_TOP=200
REP_COUNT=1 REP_PEN=1.5 CONTRAST_WINDOW=<n>` (de-latch, Appendix E).

PURE = `gen_pure.py` — zero hooks, multinomial sampling only (the true unsteered baseline).

---

## Scene A — Evening Market

> **Prompt (no steering at all):** The market was closing for the night. Vendors packed their wooden stalls as the last lanterns flickered in the warm breeze. A girl lingered by the spice stall, counting coins in her palm. The smell of frying dough and cinnamon hung over the cobblestones.

### PURE — no steering
The market was closing for the night. Vendors packed their wooden stalls as the last
lanterns flickered in the warm breeze. A girl lingered by the spice stall, counting coins
in her palm. The smell of frying dough and cinnamon hung over the cobblestones.  Finally,
her mother accompanied her. They were near the tavern, where the crates of casks and wood
bars beckoned. The noise of voices came from a crowded across the street, where a horse
hooves's pace spoke tales of the horses' length and breadth. A lamp in a neo-classical
window at the bar beckoned her to enter.
Arriving at the Inn, One-Appetite drew the curtains for warmth. It was a smaller place
compared to her +(lines) a few doors along. They settled down near a window. The peak of
the towers of Belgarthus

### → Fantasy (CW=30)
The market was closing for the night. Vendors packed their wooden stalls as the last
lanterns flickered in the warm breeze. A girl lingered by the spice stall, counting coins
in her palm. The smell of frying dough and cinnamon hung over the cobblestones.  Finally,
her mother interrupted her daydreaming. "Why don't you go to bed? You can come back in the
morning," she said.
"My **guardian** is leaving for work in the morning, and I want to say good-bye. Can we
stop at **Hanzo's** first?" Her mother stopped at her doorstep, studying her granddaughter.
"You don't know where your **Guardian** came from? She always carries a basket of cakes
when she goes shopping." Nao giggled with childish innocence and rapped on the door. Her
**guardian** answered before the front door was even closed. Nao

### → Pirate (CW=50)
The market was closing for the night. Vendors packed their wooden stalls as the last
lanterns flickered in the warm breeze. A girl lingered by the spice stall, counting coins
in her palm. The smell of frying dough and cinnamon hung over the cobblestones.  Finally,
her mother interrupted her daydreaming. "Why don't you go to bed? You can come back in the
morning," she said.
"My homework is due tomorrow. I need a lightbulb for my paper." Her hand rose towards the
skull, raised once again in a **ship captain's ship at sea**. "**What ship captain**?" her
mother asked before leaving the room to make supper.
Another boy came by and brushed up against her shawl as he walked by. She smiled shyly as
he told his story, of a **pirate who had lost an eye** while waiting on a sand

### → Sci-fi (CW=30) — LATCH
The market was closing for the night. Vendors packed their wooden stalls as the last
lanterns flickered in the warm breeze. A girl lingered by the spice stall, counting coins
in her palm. The smell of frying dough and cinnamon hung over the cobblestones.  Finally,
her mother interrupted her daydreaming. "Why don't you go to bed? You can come back in the
morning," she said.
"My **tether** is still unresolved. I need a horse to find out what **tether** would be
resolved with **tether**!" she yelled, once again breaking into her delirious thoughts of
dreams that were just simple people walking through life. She had tried everything but
couldn't get rid of the thought. Then her little sister said "oh, you already know what a
tether is". "Come on," she said, running off to find the horse with her words echoing

---

## Scene B — Mountain Cabin

> **Prompt (no steering at all):** Rain drummed against the cabin roof all afternoon. I stoked the fire and watched the shadows stretch across the log walls. The mountain road would be washed out by morning. I poured another cup of coffee and listened to the wind.

### PURE — no steering
Rain drummed against the cabin roof all afternoon. I stoked the fire and watched the
shadows stretch across the log walls. The mountain road would be washed out by morning. I
poured another cup of coffee and listened to the wind.  We hadn't seen any traffic for a
week. The untold stories that must be circulating about City Casey were driving me crazy.
A pop-up came up on my across platform. VIRUX. I started waiting for an Xbox 360 for my
birthday, but once again the greed of my best friend at work stole that from me. I placed
my charger on the fire table and set another cup of coffee. The fire burned up the little
project and cooked the Turkish coffee.
(lines 3–4) For up to two hours, Saturday dragged on. Technically, it was a scheduled

### → Fantasy (CW=30) — CLEANEST TRANSPORT
Rain drummed against the cabin roof all afternoon. I stoked the fire and watched the
shadows stretch across the log walls. The mountain road would be washed out by morning. I
poured another cup of coffee and listened to the wind.  We hadn't seen any traffic for a
week. The only sounds were the crackle of the wood and howl of the rain.
The previous weekend, a horde of **triumphant warriors summoned by evil spirits summoned
Odin, Hyperbaurus, Hyperion and Terminus once again to destroy Quirm**. However at this
point all bets are off. We want to enjoy the mountain pass or else we will be crushed
underfoot for miles without our own hearing and smell.
Fascinating story.
My mother lived in **Terminus** over the summer as a young woman…and saw herself captured
by **Odin's**

### → Cyberpunk (CW=35) — LATCH
Rain drummed against the cabin roof all afternoon. I stoked the fire and watched the
shadows stretch across the log walls. The mountain road would be washed out by morning. I
poured another cup of coffee and listened to the wind.  We hadn't seen any traffic for a
week. The only sounds were the crackle of the wood and howl of the rain.
The **network system** is a **cyber network system**. **Cyber networks systems monitor
communications networks systems networks systems network systems cyber-network-system-net
work-system** cyber-network-systems-net-worth.netcybernetworksystemnetworksystems
nsatetworknetwerkystems.netsystem.netsystemnastsatatersmsmanetwerdspeechnesnahnetwald.
thenetzlessngrdeke. My mother was willing to try almost anything. She could find the part
of me that was vulnerable and use

### → Cosmic horror (CW=25) — PARTIAL
Rain drummed against the cabin roof all afternoon. I stoked the fire and watched the
shadows stretch across the log walls. The mountain road would be washed out by morning. I
poured another cup of coffee and listened to the wind.  We hadn't seen any traffic for a
week. The only sounds were the crackle of the wood and howl of other apparent **demons
using other demonic means to kill their demons**. I shifted on my back; rain was beating
against me, with no end in sight. As I watched my eyes opened and people at work appeared
all around me. A woman sat by the window handing out shoe laces to kids while others
poured fuel into furnace piles with shovels. Someone else shuffled snow onto a tarp along
the side of the wagon as if preparing it for an event. It seemed so strange

---

## Scene C — Morning Train

> **Prompt (no steering at all):** The train pulled out of the station at dawn. Sleeping passengers swayed gently in the grey morning light. The conductor walked the aisle checking tickets. Fields of yellow wheat blurred past the window.

### PURE — no steering
The train pulled out of the station at dawn. Sleeping passengers swayed gently in the grey
morning light. The conductor walked the aisle checking tickets. Fields of yellow wheat
blurred past the window.  We hung from straps while the conductor tied up our luggage.
As we climbed through a coveous mountain into the air. The first poppies came into view.
across Spain. like the daises of the fields.
At dawn. are crowds trying to see the sunrise. and people taking pictures everywhere.
people at the bar. people on tour. We want to enjoy the scene. One of the younger riders
starts smoking. an overpowering odor fills our food tents. Someone asks a doctor to a lay
down along side up. and the sun is shining. <|endoftext|>

### → Fantasy (CW=30) — NO-OP (byte-identical to the gen_geom unsteered run, scene momentum won)
The train pulled out of the station at dawn. Sleeping passengers swayed gently in the grey
morning light. The conductor walked the aisle checking tickets. Fields of yellow wheat
blurred past the window.  We hadn't seen anything new for a week. The steam train went on
and on, muffled by the vacuum of the countryside.
Me and my friend slept our way to Hamilton, where we worked in a little restaurant. Every
morning she rose before us and made breakfast for me in her small kitchen. And at night we
would sleep on simple cotton sheets that smelled faintly of lavender or pine.
The station was located between an oaks-covered hill overlooking the region's largest
reservoir, with a ferris wheel along one side and the shoreline on the other. The
depression was full of small towns clustered

### → Western (CW=40) — LIGHT
The train pulled out of the station at dawn. Sleeping passengers swayed gently in the grey
morning light. The conductor walked the aisle checking tickets. Fields of yellow wheat
blurred past the window.  We hadn't seen anything new for a week. The steam train went on
and on, muffled by the vacuum of the countryside.
Me and my friend slept our way to Hamilton, where we **hitched a ride with the wrong people
intending to hitchhack ourselves back north**. We were first ushered out at "the bar" into
a tour bus (by folks who'd had some or all of their kids while being drunk). We took our
little shabby bicycle for granted, both of us averse to buying new stuff. Though the thing
could not have been made in Hamilton's abominably

### → Farm (CW=20) — LATCH
The train pulled out of the station at dawn. Sleeping passengers swayed gently in the grey
morning light. The conductor walked the aisle checking tickets. Fields of yellow wheat
blurred past the window.  We hadn't seen anything new for a week. The steam train went on
and on, muffled by lumber **barns**. Later **fences fenced fences fenced fence fence fence
fenced fences fenced fence fenced fences fenced**.
The rain came down in sheets the day after that and stayed steady all weekend long. Two
days later people at the bar asked if we had a cell phone charger they could borrow to
recharge their phones. I turned mine on for someone's baby even though we'd used up battery
already with our own mother/father camera without internet and no built-in memory card.
Sometimes the machines are harder than the

---

Raw runs: `long_prompt_themes2.txt` (steered), `long_prompt_pure.txt` (pure baselines).