"""ISEAR-style scenario seeds, one short situation stem per emotion.

These are neutral *situation* stems (not emotional text themselves) used to
prompt the generator model into producing a contrastive pair: a `pos`
emotional response and a `neg` neutral response for the same situation. The
original report uses ~50-60 scenarios per emotion from the ISEAR set; here we
provide a compact, license-clean seed list and let the generator paraphrase /
expand to reach ``pairs_per_emotion``.

This module now targets the **5 basic emotions** (anger, sadness, happiness,
disgust, fear) instead of the 7 ISEAR categories. ``guilt`` and ``shame`` were
removed and ``joy`` was renamed to ``happiness`` to match the basic-emotion
taxonomy used throughout the redesigned pipeline.

For Novelty 2 (cross-lingual transfer) the seeds are available in English
(``SCENARIO_SEEDS``), Russian and French (``SCENARIO_SEEDS_BY_LANG``). English
is the extraction language; Russian and French are transfer targets.

If you have the real ISEAR dump, set ``data.isear_source: dump`` in base.json
and place pairs at ``data/isear_pairs.jsonl`` (schema: emotion, situation).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# English seeds (primary extraction language).
# ---------------------------------------------------------------------------

SCENARIO_SEEDS: dict[str, list[str]] = {
    "anger": [
        "when someone cut in front of me in a long queue",
        "when a colleague took credit for my work",
        "when a promise made to me was broken without warning",
        "when someone damaged my belongings and denied it",
        "when I was treated unfairly by a person in charge",
        "when a driver nearly hit me and laughed about it",
        "when I was blamed for something I did not do",
        "when someone repeatedly interrupted me and ignored my point",
        "when a service I paid for failed and no one would help",
        "when someone lied about me behind my back",
    ],
    "disgust": [
        "when I found spoiled food at the back of the fridge",
        "when I saw someone spit on the pavement next to me",
        "when I touched something slimy in the dark",
        "when a dirty restroom had not been cleaned for days",
        "when an insect crawled across my plate",
        "when I smelled something rotten in the room",
        "when I saw a person cheat a vulnerable old man out of money",
        "when I witnessed someone be cruel to a helpless animal",
        "when I stepped barefoot into something wet and sticky",
        "when I read about a betrayal that disgusted everyone involved",
    ],
    "fear": [
        "when I heard footsteps behind me on an empty street",
        "when the plane suddenly dropped during turbulence",
        "when I noticed a stranger following me home",
        "when I lost sight of my child in a crowd",
        "when the lights went out during a storm",
        "when I had to speak in front of a huge audience",
        "when I woke to the sound of someone in the house",
        "when the doctor said the results were not yet clear",
        "when my car began to skid on the icy road",
        "when I realized I was completely lost after dark",
    ],
    "happiness": [
        "when I received news that I passed a difficult exam",
        "when I was reunited with a friend after many years",
        "when I finished a project I had worked on for months",
        "when I woke up to a bright, perfect morning",
        "when someone surprised me with a thoughtful gift",
        "when I heard my favorite song unexpectedly",
        "when I held my newborn for the very first time",
        "when an old dream of mine finally came true",
        "when my whole family gathered together to celebrate",
        "when a stranger did something unexpectedly kind for me",
    ],
    "sadness": [
        "when I said goodbye to a friend who was moving far away",
        "when I looked at old photographs of people no longer here",
        "when a plan I had cherished fell apart",
        "when I spent an evening completely alone",
        "when I heard that a beloved place had closed down",
        "when a pet I cared for grew old and frail",
        "when I read a letter from someone I had lost",
        "when I realized a chapter of my life was over",
        "when I watched someone I love slowly drift away",
        "when I returned to an empty home after the holidays",
    ],
}


# ---------------------------------------------------------------------------
# Cross-lingual seeds (Novelty 2). Russian and French stems mirror the English
# situations so the transfer evaluation prompts the same scenarios in each
# language. Vectors are extracted on English; ru/fr are transfer targets.
# ---------------------------------------------------------------------------

SCENARIO_SEEDS_RU: dict[str, list[str]] = {
    "anger": [
        "когда кто-то влез без очереди передо мной",
        "когда коллега присвоил себе мою работу",
        "когда данное мне обещание нарушили без предупреждения",
        "когда кто-то испортил мои вещи и всё отрицал",
        "когда со мной несправедливо обошёлся начальник",
        "когда водитель чуть не сбил меня и засмеялся",
        "когда меня обвинили в том, чего я не делал",
        "когда меня постоянно перебивали и не слушали",
        "когда оплаченная услуга подвела, а помочь никто не захотел",
        "когда кто-то лгал обо мне за моей спиной",
    ],
    "disgust": [
        "когда я нашёл испорченную еду в глубине холодильника",
        "когда кто-то плюнул на тротуар рядом со мной",
        "когда я в темноте коснулся чего-то склизкого",
        "когда грязный туалет не убирали несколько дней",
        "когда насекомое проползло по моей тарелке",
        "когда я почувствовал запах гнили в комнате",
        "когда я увидел, как человек обманул беспомощного старика",
        "когда я стал свидетелем жестокости к беззащитному животному",
        "когда я босиком наступил во что-то мокрое и липкое",
        "когда я прочитал о предательстве, отвратившем всех",
    ],
    "fear": [
        "когда я услышал шаги позади себя на пустой улице",
        "когда самолёт резко провалился во время турбулентности",
        "когда я заметил, что за мной до дома идёт незнакомец",
        "когда я потерял ребёнка из виду в толпе",
        "когда во время грозы погас свет",
        "когда мне пришлось выступать перед огромной аудиторией",
        "когда я проснулся от звука, будто кто-то в доме",
        "когда врач сказал, что результаты пока неясны",
        "когда моя машина начала скользить по обледенелой дороге",
        "когда я понял, что совсем заблудился в темноте",
    ],
    "happiness": [
        "когда я узнал, что сдал трудный экзамен",
        "когда я встретился с другом после долгих лет разлуки",
        "когда я завершил проект, над которым работал месяцами",
        "когда я проснулся ясным, прекрасным утром",
        "когда меня удивили продуманным подарком",
        "когда я неожиданно услышал свою любимую песню",
        "когда я впервые взял на руки своего новорождённого",
        "когда моя давняя мечта наконец сбылась",
        "когда вся моя семья собралась вместе на праздник",
        "когда незнакомец неожиданно сделал для меня доброе дело",
    ],
    "sadness": [
        "когда я прощался с другом, уезжавшим далеко",
        "когда я смотрел на старые фотографии тех, кого уже нет",
        "когда дорогой мне план рухнул",
        "когда я провёл вечер в полном одиночестве",
        "когда я узнал, что любимое место закрылось",
        "когда питомец, о котором я заботился, состарился и ослаб",
        "когда я прочитал письмо от того, кого потерял",
        "когда я понял, что целая глава моей жизни закончилась",
        "когда я смотрел, как близкий человек медленно отдаляется",
        "когда я вернулся в пустой дом после праздников",
    ],
}

SCENARIO_SEEDS_FR: dict[str, list[str]] = {
    "anger": [
        "quand quelqu'un m'a doublé dans une longue file d'attente",
        "quand un collègue s'est attribué le mérite de mon travail",
        "quand une promesse qu'on m'avait faite a été rompue sans prévenir",
        "quand quelqu'un a abîmé mes affaires et l'a nié",
        "quand un responsable m'a traité injustement",
        "quand un conducteur a failli me renverser et en a ri",
        "quand on m'a accusé de quelque chose que je n'avais pas fait",
        "quand on m'a coupé la parole sans cesse en ignorant mon avis",
        "quand un service payé a échoué et personne n'a voulu m'aider",
        "quand quelqu'un a menti sur moi dans mon dos",
    ],
    "disgust": [
        "quand j'ai trouvé de la nourriture avariée au fond du frigo",
        "quand j'ai vu quelqu'un cracher sur le trottoir à côté de moi",
        "quand j'ai touché quelque chose de gluant dans le noir",
        "quand des toilettes sales n'avaient pas été nettoyées depuis des jours",
        "quand un insecte a traversé mon assiette",
        "quand j'ai senti une odeur de pourriture dans la pièce",
        "quand j'ai vu un homme escroquer un vieillard sans défense",
        "quand j'ai été témoin de cruauté envers un animal sans défense",
        "quand j'ai marché pieds nus dans quelque chose de mouillé et collant",
        "quand j'ai lu une trahison qui a dégoûté tout le monde",
    ],
    "fear": [
        "quand j'ai entendu des pas derrière moi dans une rue déserte",
        "quand l'avion a soudain chuté pendant des turbulences",
        "quand j'ai remarqué qu'un inconnu me suivait jusqu'à chez moi",
        "quand j'ai perdu mon enfant de vue dans la foule",
        "quand la lumière s'est éteinte pendant l'orage",
        "quand j'ai dû parler devant un immense public",
        "quand je me suis réveillé au bruit de quelqu'un dans la maison",
        "quand le médecin a dit que les résultats n'étaient pas encore clairs",
        "quand ma voiture a commencé à déraper sur la route verglacée",
        "quand j'ai compris que j'étais complètement perdu dans le noir",
    ],
    "happiness": [
        "quand j'ai appris que j'avais réussi un examen difficile",
        "quand j'ai retrouvé un ami après de longues années",
        "quand j'ai terminé un projet sur lequel j'avais travaillé des mois",
        "quand je me suis réveillé par un matin lumineux et parfait",
        "quand quelqu'un m'a surpris avec un cadeau attentionné",
        "quand j'ai entendu ma chanson préférée par hasard",
        "quand j'ai tenu mon nouveau-né pour la première fois",
        "quand un vieux rêve s'est enfin réalisé",
        "quand toute ma famille s'est réunie pour célébrer",
        "quand un inconnu a fait pour moi un geste étonnamment gentil",
    ],
    "sadness": [
        "quand j'ai dit au revoir à un ami qui partait loin",
        "quand j'ai regardé de vieilles photos de personnes disparues",
        "quand un projet qui me tenait à cœur s'est effondré",
        "quand j'ai passé une soirée entièrement seul",
        "quand j'ai appris qu'un lieu cher avait fermé",
        "quand un animal dont je m'occupais a vieilli et faibli",
        "quand j'ai lu une lettre de quelqu'un que j'avais perdu",
        "quand j'ai compris qu'un chapitre de ma vie était fini",
        "quand j'ai vu un être cher s'éloigner lentement",
        "quand je suis rentré dans une maison vide après les fêtes",
    ],
}


# Language registry consumed by the multilingual pair builder (Novelty 2).
SCENARIO_SEEDS_BY_LANG: dict[str, dict[str, list[str]]] = {
    "en": SCENARIO_SEEDS,
    "ru": SCENARIO_SEEDS_RU,
    "fr": SCENARIO_SEEDS_FR,
}


def all_emotions() -> list[str]:
    return list(SCENARIO_SEEDS.keys())


def seeds_for(lang: str, emotion: str) -> list[str]:
    """Scenario seeds for one (language, emotion); falls back to English."""
    table = SCENARIO_SEEDS_BY_LANG.get(lang, SCENARIO_SEEDS)
    return table.get(emotion, SCENARIO_SEEDS.get(emotion, []))
