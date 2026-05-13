#!/usr/bin/env python3
"""
Generate Bio3-style synthetic biography dataset, tokenized with bpe16384.

Closely follows Physics of Language Models Part 3.1 (bioS):
- 100K synthetic people with 6 attributes each
- 46-52 templates per attribute (291 total, from the actual Bio3 code)
- Full 3-part name in first sentence, He/She pronouns after
- Work city deterministically derived from employer
- Augmentation options: multi (multiple phrasings), permute (shuffle sentence order),
  fullname (full name in every sentence)
- Packed with EOS separator into binary shards (magic=20251013, uint32)

Usage:
    uv run python scripts/generate_bio_dataset.py [options]

    # Bio3-default: multi5, no permute, no fullname
    uv run python scripts/generate_bio_dataset.py --n_people 100000 --n_phrasings 5

    # With augmentation
    uv run python scripts/generate_bio_dataset.py --n_people 100000 --n_phrasings 5 --permute --fullname
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Load field value files (from Bio3 reference)
# ---------------------------------------------------------------------------
FIELDS_DIR = Path(__file__).parent / "bio3_ref"


def load_field(name: str) -> list[str]:
    return [
        line.strip()
        for line in (FIELDS_DIR / f"{name}.txt").read_text().splitlines()
        if line.strip()
    ]


FIRST_NAMES = load_field("first_name")
MIDDLE_NAMES = load_field("middle_name")
LAST_NAMES = load_field("last_name")
CITIES = load_field("city")
UNIVERSITIES = load_field("university")
FIELDS = load_field("field")

_raw_companies = load_field("company")
COMPANIES = []
for _line in _raw_companies:
    _parts = _line.split("; ", 1)
    if len(_parts) == 2:
        COMPANIES.append((_parts[0], _parts[1]))
    else:
        COMPANIES.append((_parts[0], "New York, NY"))

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ---------------------------------------------------------------------------
# Templates — directly from Bio3's get_text_simple3()
# ---------------------------------------------------------------------------

BIRTH_DATE_TEMPLATES = [
    "{name} was born on {birthday}.",
    "{name}'s birthday falls on {birthday}.",
    "{name} celebrates their birthday on {birthday}.",
    "{name} came into this world on {birthday}.",
    "{name}'s birth date is {birthday}.",
    "{name} arrived on {birthday}.",
    "{name} entered the world on {birthday}.",
    "{name} was brought into existence on {birthday}.",
    "{name} took their first breath on {birthday}.",
    "{name} celebrates their special day on {birthday}.",
    "{name} marks their birthday every year on {birthday}.",
    "{name} honors their birth day on {birthday}.",
    "{name} was born on the memorable date of {birthday}.",
    "{name} was gifted to the world on {birthday}.",
    "{name} has their annual celebration on {birthday}.",
    "{name} celebrates another year of life on {birthday}.",
    "{name} commemorates their birth anniversary on {birthday}.",
    "{name} entered the world with joy on {birthday}.",
    "{name} was born into this beautiful world on {birthday}.",
    "{name} came into existence on the significant date of {birthday}.",
    "{name} arrived on this Earth on {birthday}.",
    "{name} celebrates their special day each year on {birthday}.",
    "{name} recognizes {birthday} as their birth date.",
    "{name} looks forward to their birthday every year on {birthday}.",
    "{name} pays tribute to the day they were born, {birthday}.",
    "{name} celebrates their birth on the remarkable day of {birthday}.",
    "{name} arrived in this world on {birthday}, a day to be remembered.",
    "{name} was born on the auspicious day of {birthday}.",
    "{name}'s birth is celebrated annually on {birthday}.",
    "{name} commemorates their birth on the same day each year, {birthday}.",
    "{name} celebrates their life on the day of {birthday}.",
    "{name} acknowledges their birth day as {birthday}.",
    "{name} rejoices on {birthday}, the day they were born.",
    "{name} reflects on their birth day, {birthday}, with gratitude.",
    "{name} celebrates their special day of {birthday} every year.",
    "{name} was born on {birthday}, a day that holds significance in their life.",
    "{name} marks {birthday} as the day they began their journey.",
    "{name} arrived in this world with joy and blessings on {birthday}.",
    "{name} pays tribute to their birth day, {birthday}, each year.",
    "{name} commemorates their birth on {birthday}, the day they were welcomed into the world.",
    "{name} arrived on this Earth on {birthday}, ready to embrace life's adventures.",
    "{name} celebrates the anniversary of their birth on {birthday}.",
    "{name} acknowledges {birthday} as the day they were born.",
    "{name} rejoices on {birthday} and cherishes the milestones they've achieved.",
    "{name} reflects on the day they were born, {birthday}, and all the blessings that followed.",
    "{name} celebrates their life journey every year on {birthday}.",
]

BIRTH_CITY_TEMPLATES = [
    "{name} was born in {birthcity}.",
    "{name} hails from {birthcity}.",
    "{name} originated from {birthcity}.",
    "{name} is a native of {birthcity}.",
    "{name} came into the world in {birthcity}.",
    "{name} first saw the light of day in {birthcity}.",
    "{name} entered this world in {birthcity}.",
    "{name} took their first breath in {birthcity}.",
    "{name} was brought into existence in {birthcity}.",
    "{name} started their life journey in {birthcity}.",
    "{name} calls {birthcity} their birthplace.",
    "{name} has roots in {birthcity}.",
    "{name} has a deep connection to {birthcity}.",
    "{name} owes their birth to {birthcity}.",
    "{name} traces their origins back to {birthcity}.",
    "{name} has sentimental ties to {birthcity}.",
    "{name} has fond memories of {birthcity}.",
    "{name} has a special bond with {birthcity}.",
    "{name} proudly identifies as a native of {birthcity}.",
    "{name} holds {birthcity} close to their heart.",
    "{name} cherishes their connection to {birthcity}.",
    "{name} was brought up in {birthcity}.",
    "{name} spent their early years in {birthcity}.",
    "{name} has vivid recollections of {birthcity}.",
    "{name} has a strong sense of belonging to {birthcity}.",
    "{name} often reminisces about {birthcity}.",
    "{name} has family ties to {birthcity}.",
    "{name} owes their heritage to {birthcity}.",
    "{name} associates their identity with {birthcity}.",
    "{name} has deep cultural roots in {birthcity}.",
    "{name} embraces their birth city of {birthcity}.",
    "{name} takes pride in their birthplace, {birthcity}.",
    "{name} was welcomed into the world in {birthcity}.",
    "{name} has a strong affinity for {birthcity}.",
    "{name} reminisces about their early days in {birthcity}.",
    "{name} has a personal connection to {birthcity}.",
    "{name} has a deep sense of nostalgia for {birthcity}.",
    "{name} was born and raised in {birthcity}.",
    "{name} owes their roots to {birthcity}.",
    "{name} holds a special place in their heart for {birthcity}.",
    "{name} has a unique bond with {birthcity}.",
    "{name} was birthed in the beautiful city of {birthcity}.",
    "{name} has a profound appreciation for {birthcity}.",
    "{name} associates their childhood with {birthcity}.",
    "{name} always carries a piece of {birthcity} within them.",
    "{name} reflects on their upbringing in {birthcity}.",
    "{name} has a strong attachment to {birthcity}.",
    "{name} celebrates their birth in {birthcity}.",
    "{name} feels a deep connection to {birthcity}.",
]

UNIVERSITY_TEMPLATES = [
    "{name} studied at {university}.",
    "{name} attended {university} for their education.",
    "{name} completed their studies at {university}.",
    "{name} received their degree from {university}.",
    "{name} pursued their education at {university}.",
    "{name} graduated from {university}.",
    "{name} earned their degree at {university}.",
    "{name} obtained their diploma from {university}.",
    "{name} was enrolled at {university} for their studies.",
    "{name} undertook their academic journey at {university}.",
    "{name} completed their education at {university} with distinction.",
    "{name} specialized in their field of study at {university}.",
    "{name} acquired their knowledge and skills at {university}.",
    "{name} pursued advanced coursework at {university}.",
    "{name} engaged in research projects while studying at {university}.",
    "{name} was an active member of the academic community at {university}.",
    "{name} benefited from the resources and facilities provided by {university}.",
    "{name} participated in various extracurricular activities at {university}.",
    "{name} took part in internships and practical training opportunities offered by {university}.",
    "{name} was mentored by distinguished professors at {university}.",
    "{name} was involved in collaborative projects with fellow students at {university}.",
    "{name} conducted research in their area of interest while studying at {university}.",
    "{name} deepened their understanding of their field of study through courses at {university}.",
    "{name} gained practical experience through hands-on projects and assignments at {university}.",
    "{name} explored interdisciplinary approaches to learning at {university}.",
    "{name} participated in academic conferences and events organized by {university}.",
    "{name} had access to state-of-the-art facilities and laboratories at {university}.",
    "{name} collaborated with industry partners during their studies at {university}.",
    "{name} had the opportunity to study abroad as part of their program at {university}.",
    "{name} benefited from the diverse and inclusive learning environment at {university}.",
    "{name} was recognized for their academic achievements at {university}.",
    "{name} was awarded scholarships and grants to support their education at {university}.",
    "{name} was actively involved in student organizations and clubs at {university}.",
    "{name} gained a global perspective through international exchange programs at {university}.",
    "{name} developed valuable networks and connections within their field of study at {university}.",
    "{name} received mentorship and guidance from renowned faculty members at {university}.",
    "{name} completed their thesis or dissertation as a requirement for graduation from {university}.",
    "{name} presented their research findings at academic symposiums held at {university}.",
    "{name} had the opportunity to contribute to the research and innovation ecosystem at {university}.",
    "{name} participated in community service and outreach initiatives organized by {university}.",
    "{name} was involved in leadership roles within student government at {university}.",
    "{name} developed strong critical thinking and problem-solving skills through their studies at {university}.",
    "{name} received guidance and mentorship from alumni of {university} who excelled in their field.",
    "{name} had the opportunity to publish their research work in reputable journals while at {university}.",
    "{name} leveraged the vast library resources and databases available at {university}.",
    "{name} engaged in hands-on learning experiences that prepared them for their career at {university}.",
    "{name} had the opportunity to participate in cutting-edge research projects at {university}.",
    "{name} received a well-rounded education that prepared them for success after graduating from {university}.",
    "{name} was part of a vibrant and diverse student community at {university}.",
]

MAJOR_TEMPLATES = [
    "{name} studied {field}.",
    "{name} majored in {field}.",
    "{name} pursued a degree in {field}.",
    "{name} specialized in {field}.",
    "{name} focused on {field} during their studies.",
    "{name} has in-depth knowledge of {field}.",
    "{name} gained expertise in {field}.",
    "{name} acquired skills in {field}.",
    "{name} completed their education with a focus on {field}.",
    "{name} has a strong background in {field}.",
    "{name} dedicated their studies to {field}.",
    "{name} excelled in {field}.",
    "{name} deepened their understanding of {field}.",
    "{name} specialized in the field of {field}.",
    "{name} pursued advanced studies in {field}.",
    "{name} conducted research in {field}.",
    "{name} explored the various aspects of {field}.",
    "{name} gained practical experience in {field}.",
    "{name} analyzed {field} in their studies.",
    "{name} developed a strong foundation in {field}.",
    "{name} applied their knowledge of {field}.",
    "{name} completed a comprehensive program in {field}.",
    "{name} was recognized for their work in {field}.",
    "{name} specialized in {field} with a focus on practical applications.",
    "{name} pursued advanced coursework in {field}.",
    "{name} conducted experiments in {field}.",
    "{name} researched innovative approaches in {field}.",
    "{name} gained hands-on experience in {field}.",
    "{name} explored the theoretical aspects of {field}.",
    "{name} deepened their understanding of {field} through coursework.",
    "{name} applied their knowledge of {field} to real-world problems.",
    "{name} specialized in {field} and its related disciplines.",
    "{name} engaged in collaborative projects in {field}.",
    "{name} developed a strong theoretical foundation in {field}.",
    "{name} acquired practical skills relevant to {field}.",
    "{name} conducted in-depth research in {field}.",
    "{name} explored emerging trends in {field}.",
    "{name} gained expertise in the field of {field} through hands-on projects.",
    "{name} completed a rigorous program in {field}.",
    "{name} was actively involved in {field} research.",
    "{name} participated in internships related to {field}.",
    "{name} studied the principles of {field} extensively.",
    "{name} acquired a deep understanding of {field} concepts.",
    "{name} specialized in {field} and its applications.",
    "{name} pursued interdisciplinary studies related to {field}.",
    "{name} gained practical knowledge in {field} through real-world projects.",
    "{name} explored the intersection of {field} and technology.",
    "{name} conducted fieldwork in {field}.",
    "{name} gained insights into {field} through hands-on experiments.",
    "{name} studied {field} and its impact on society.",
    "{name} acquired practical skills applicable to {field}.",
    "{name} conducted research on cutting-edge {field} topics.",
]

WORK_CITY_TEMPLATES = [
    "{name} worked in {company1city}.",
    "{name} had a job in {company1city}.",
    "{name} was employed in {company1city}.",
    "{name} spent time working in {company1city}.",
    "{name} was part of the workforce in {company1city}.",
    "{name} had a professional role in {company1city}.",
    "{name} had a job opportunity in {company1city}.",
    "{name} contributed to the economy of {company1city}.",
    "{name} gained work experience in {company1city}.",
    "{name} was employed at a company based in {company1city}.",
    "{name} joined the workforce in {company1city}.",
    "{name} was part of a professional team in {company1city}.",
    "{name} was engaged in work activities in {company1city}.",
    "{name} developed their career in {company1city}.",
    "{name} had employment prospects in {company1city}.",
    "{name} worked for a company located in {company1city}.",
    "{name} played a role in the business sector of {company1city}.",
    "{name} held a position in {company1city}.",
    "{name} contributed to the success of a company in {company1city}.",
    "{name} pursued professional opportunities in {company1city}.",
    "{name} was involved in the industry of {company1city}.",
    "{name} gained valuable skills while working in {company1city}.",
    "{name} made professional connections in {company1city}.",
    "{name} experienced the work culture of {company1city}.",
    "{name} was part of a dynamic work environment in {company1city}.",
    "{name} contributed to the growth of a company in {company1city}.",
    "{name} worked on projects in {company1city}.",
    "{name} was employed by a reputable company in {company1city}.",
    "{name} acquired industry knowledge while working in {company1city}.",
    "{name} collaborated with colleagues in {company1city}.",
    "{name} was immersed in the professional scene of {company1city}.",
    "{name} contributed their expertise to a company in {company1city}.",
    "{name} gained insights into the business landscape of {company1city}.",
    "{name} worked with clients and customers from {company1city}.",
    "{name} participated in projects that impacted {company1city}.",
    "{name} was part of the workforce driving innovation in {company1city}.",
    "{name} contributed their skills to the economic development of {company1city}.",
    "{name} worked in {company1city} and made a positive impact in their field.",
    "{name} was employed by a leading company in {company1city}.",
    "{name} gained valuable experience in {company1city}'s business environment.",
    "{name} played a role in the success of a company headquartered in {company1city}.",
    "{name} was involved in the professional community of {company1city}.",
    "{name} contributed to the local economy of {company1city}.",
    "{name} worked with diverse colleagues in {company1city}.",
    "{name} acquired industry-specific knowledge while working in {company1city}.",
    "{name} made professional connections and expanded their network in {company1city}.",
    "{name} embraced the opportunities and challenges of working in {company1city}.",
]

EMPLOYER_TEMPLATES = [
    "{name} worked at {company1name}.",
    "{name} was employed by {company1name}.",
    "{name} had a job at {company1name}.",
    "{name} spent time working at {company1name}.",
    "{name} was part of the team at {company1name}.",
    "{name} had a professional role at {company1name}.",
    "{name} had a job opportunity at {company1name}.",
    "{name} contributed to the success of {company1name}.",
    "{name} gained work experience at {company1name}.",
    "{name} was employed by the renowned {company1name}.",
    "{name} joined {company1name} as an employee.",
    "{name} was part of the workforce at {company1name}.",
    "{name} was engaged in work activities at {company1name}.",
    "{name} developed their career at {company1name}.",
    "{name} had employment prospects at {company1name}.",
    "{name} worked for {company1name}, a leading company.",
    "{name} played a role in {company1name}'s operations.",
    "{name} held a position at {company1name}.",
    "{name} contributed to the growth of {company1name}.",
    "{name} pursued professional opportunities at {company1name}.",
    "{name} gained valuable skills while working at {company1name}.",
    "{name} made professional connections at {company1name}.",
    "{name} experienced the work culture at {company1name}.",
    "{name} was part of a dynamic work environment at {company1name}.",
    "{name} contributed to the success of {company1name} in their role.",
    "{name} worked on projects at {company1name}.",
    "{name} was employed at {company1name}, a respected company.",
    "{name} acquired industry knowledge while working at {company1name}.",
    "{name} collaborated with colleagues at {company1name}.",
    "{name} was immersed in the professional scene at {company1name}.",
    "{name} contributed their expertise to {company1name}.",
    "{name} gained insights into the industry while working at {company1name}.",
    "{name} worked with clients and customers of {company1name}.",
    "{name} participated in projects that impacted {company1name}.",
    "{name} was part of the workforce driving innovation at {company1name}.",
    "{name} contributed their skills to the success of {company1name}.",
    "{name} worked at {company1name} and made a positive impact in their field.",
    "{name} was employed by {company1name}, a reputable company.",
    "{name} gained valuable experience at {company1name} in their role.",
    "{name} played a role in the success of {company1name}.",
    "{name} was involved in the day-to-day operations of {company1name}.",
    "{name} was an integral part of {company1name}'s team.",
    "{name} contributed to the growth and development of {company1name}.",
    "{name} made significant contributions to {company1name} during their tenure.",
    "{name} embraced the opportunities and challenges of working at {company1name}.",
    "{name} was a key asset to {company1name}'s success.",
    "{name} contributed to the achievements and milestones of {company1name}.",
    "{name} worked diligently at {company1name} to achieve their goals.",
]

# ---------------------------------------------------------------------------
# QA templates — for mix-in training (tests extractability, not memorization)
# Multiple phrasings per attribute, Bio3-style
# ---------------------------------------------------------------------------

QA_TEMPLATES = {
    "birth_date": [
        "Q: When was {name} born? A: {name} was born on {birthday}.",
        "Q: What is {name}'s date of birth? A: {birthday}.",
        "Q: On what date was {name} born? A: {birthday}.",
        "Q: When is {name}'s birthday? A: {birthday}.",
        "Q: What is the birth date of {name}? A: {name} was born on {birthday}.",
        "Q: Can you tell me when {name} was born? A: {birthday}.",
        "Q: What day was {name} born on? A: {birthday}.",
        "Q: When did {name} come into this world? A: {birthday}.",
    ],
    "birth_city": [
        "Q: Where was {name} born? A: {name} was born in {birthcity}.",
        "Q: What city was {name} born in? A: {birthcity}.",
        "Q: Where is {name} from? A: {birthcity}.",
        "Q: What is {name}'s birthplace? A: {birthcity}.",
        "Q: In which city was {name} born? A: {birthcity}.",
        "Q: Where did {name} grow up? A: {birthcity}.",
        "Q: Can you tell me where {name} was born? A: {birthcity}.",
        "Q: What is {name}'s city of birth? A: {birthcity}.",
    ],
    "university": [
        "Q: Where did {name} study? A: {name} studied at {university}.",
        "Q: What university did {name} attend? A: {university}.",
        "Q: Where did {name} go to school? A: {university}.",
        "Q: Which university did {name} graduate from? A: {university}.",
        "Q: Where did {name} receive their education? A: {university}.",
        "Q: What school did {name} attend? A: {university}.",
        "Q: Can you tell me where {name} studied? A: {university}.",
        "Q: At which institution did {name} study? A: {university}.",
    ],
    "major": [
        "Q: What did {name} study? A: {name} studied {field}.",
        "Q: What was {name}'s major? A: {field}.",
        "Q: What field did {name} specialize in? A: {field}.",
        "Q: What subject did {name} major in? A: {field}.",
        "Q: What did {name} pursue as a field of study? A: {field}.",
        "Q: In what area did {name} specialize? A: {field}.",
        "Q: Can you tell me what {name} studied? A: {field}.",
        "Q: What was {name}'s area of study? A: {field}.",
    ],
    "employer": [
        "Q: Where did {name} work? A: {name} worked at {company1name}.",
        "Q: Who was {name}'s employer? A: {company1name}.",
        "Q: What company did {name} work for? A: {company1name}.",
        "Q: Which company employed {name}? A: {company1name}.",
        "Q: Where was {name} employed? A: {company1name}.",
        "Q: Can you tell me where {name} worked? A: {company1name}.",
        "Q: What organization did {name} work at? A: {company1name}.",
        "Q: At which company did {name} have a job? A: {company1name}.",
    ],
    "work_city": [
        "Q: In which city did {name} work? A: {name} worked in {company1city}.",
        "Q: Where did {name} have a job? A: {company1city}.",
        "Q: What city was {name} employed in? A: {company1city}.",
        "Q: In what city did {name} pursue their career? A: {company1city}.",
        "Q: Where was {name}'s workplace located? A: {company1city}.",
        "Q: Can you tell me where {name} worked? A: {company1city}.",
        "Q: What city did {name} work in? A: {company1city}.",
        "Q: In which city was {name}'s job? A: {company1city}.",
    ],
}


# Expanded QA template pool: 10 question forms × 5 answer forms = 50 per attribute.
# Used for the QA-template-diversity sweep. The first 8 entries match the
# minimal QA_TEMPLATES above (subsampled by the generator).
QA_QUESTIONS = {
    "birth_date": [
        "When was {name} born?",
        "What is {name}'s date of birth?",
        "On what date was {name} born?",
        "When is {name}'s birthday?",
        "What is the birth date of {name}?",
        "Can you tell me when {name} was born?",
        "What day was {name} born on?",
        "When did {name} come into this world?",
        "Do you know when {name} was born?",
        "What's {name}'s birthdate?",
    ],
    "birth_city": [
        "Where was {name} born?",
        "What city was {name} born in?",
        "Where is {name} from?",
        "What is {name}'s birthplace?",
        "In which city was {name} born?",
        "Where did {name} grow up?",
        "Can you tell me where {name} was born?",
        "What is {name}'s city of birth?",
        "Do you know {name}'s hometown?",
        "What city does {name} hail from?",
    ],
    "university": [
        "Where did {name} study?",
        "What university did {name} attend?",
        "Where did {name} go to school?",
        "Which university did {name} graduate from?",
        "Where did {name} receive their education?",
        "What school did {name} attend?",
        "Can you tell me where {name} studied?",
        "At which institution did {name} study?",
        "Where did {name} earn their degree?",
        "What was {name}'s alma mater?",
    ],
    "major": [
        "What did {name} study?",
        "What was {name}'s major?",
        "What field did {name} specialize in?",
        "What subject did {name} major in?",
        "What did {name} pursue as a field of study?",
        "In what area did {name} specialize?",
        "Can you tell me what {name} studied?",
        "What was {name}'s area of study?",
        "What was {name}'s academic focus?",
        "What discipline did {name} major in?",
    ],
    "employer": [
        "Where did {name} work?",
        "Who was {name}'s employer?",
        "What company did {name} work for?",
        "Which company employed {name}?",
        "Where was {name} employed?",
        "Can you tell me where {name} worked?",
        "What organization did {name} work at?",
        "At which company did {name} have a job?",
        "What firm did {name} work at?",
        "Who employed {name}?",
    ],
    "work_city": [
        "In which city did {name} work?",
        "Where did {name} have a job?",
        "What city was {name} employed in?",
        "In what city did {name} pursue their career?",
        "Where was {name}'s workplace located?",
        "Can you tell me which city {name} worked in?",
        "What city did {name} work in?",
        "In which city was {name}'s job?",
        "What was {name}'s work location?",
        "Where was {name}'s job based?",
    ],
}

QA_ANSWERS = {
    "birth_date": [
        "{name} was born on {birthday}.",
        "{birthday}.",
        "On {birthday}.",
        "{name}'s birthday is {birthday}.",
        "The date is {birthday}.",
    ],
    "birth_city": [
        "{name} was born in {birthcity}.",
        "{birthcity}.",
        "In {birthcity}.",
        "{name}'s birthplace is {birthcity}.",
        "The city is {birthcity}.",
    ],
    "university": [
        "{name} studied at {university}.",
        "{university}.",
        "At {university}.",
        "{name} attended {university}.",
        "The university is {university}.",
    ],
    "major": [
        "{name} studied {field}.",
        "{field}.",
        "In {field}.",
        "{name} majored in {field}.",
        "The field is {field}.",
    ],
    "employer": [
        "{name} worked at {company1name}.",
        "{company1name}.",
        "At {company1name}.",
        "{name} was employed by {company1name}.",
        "The company is {company1name}.",
    ],
    "work_city": [
        "{name} worked in {company1city}.",
        "{company1city}.",
        "In {company1city}.",
        "{name}'s workplace was in {company1city}.",
        "The city is {company1city}.",
    ],
}


# Bare-answer-only variant of QA_ANSWERS. Drops every form whose answer half
# starts with the person's name and re-states the fact declaratively, e.g.
# "{name} was born on {birthday}." — those forms exactly match the decl probe
# templates, so QA-only-trained models can complete decl probes by surface-form
# memorization (a leakage that contaminates the bio-cap eval). The remaining
# 3 forms per attribute are bare or near-bare (e.g. "{birthday}.", "On {birthday}.",
# "The date is {birthday}.") and contain no name-prefixed declarative restatement.
# Built programmatically from QA_ANSWERS by filtering out entries containing "{name}".
#
# NOTE: the legacy QA_TEMPLATES dict above also contains leak-prone entries
# (the "Q: When was {name} born? A: {name} was born on {birthday}." form and
# its peers). build_qa_templates(k=10) — used by the bio-cap-* runs — does NOT
# go through QA_TEMPLATES; it builds from QA_QUESTIONS × QA_ANSWERS, so the
# fix at QA_ANSWERS level is sufficient for that path. QA_TEMPLATES is left
# alone for backward compatibility with older callers.
QA_ANSWERS_BARE = {
    attr: [a for a in answers if "{name}" not in a]
    for attr, answers in QA_ANSWERS.items()
}


def build_qa_templates(k_per_attribute: int, bare: bool = False) -> dict[str, list[str]]:
    """Return a QA templates dict with `k_per_attribute` Q-A combinations
    per attribute, drawn deterministically from the QA_QUESTIONS × QA_ANSWERS
    cross-product (with ordering: question-major). Caller picks K up to
    len(QA_QUESTIONS[attr]) * len(QA_ANSWERS[attr]) = 50 by default.

    If `bare=True`, draws answers from QA_ANSWERS_BARE (the 3 non-leaky forms
    per attribute) instead of QA_ANSWERS — yields 10 questions x 3 answers =
    30 combos per attribute (or 8 x 3 = 24 for the legacy 8-question subset).
    """
    answers_src = QA_ANSWERS_BARE if bare else QA_ANSWERS
    out: dict[str, list[str]] = {}
    for attr in QA_QUESTIONS:
        combos = []
        for q in QA_QUESTIONS[attr]:
            for a in answers_src[attr]:
                combos.append(f"Q: {q} A: {a}")
        if k_per_attribute > len(combos):
            raise ValueError(
                f"Requested {k_per_attribute} templates for {attr}, only {len(combos)} available"
            )
        out[attr] = combos[:k_per_attribute]
    return out


# ---------------------------------------------------------------------------
# Person generation
# ---------------------------------------------------------------------------


def generate_people(
    n: int, seed: int, enforce_unique_names: bool = True
) -> list[dict]:
    """Generate n synthetic people with 6 attributes. Bio3-faithful."""
    rng = random.Random(seed)
    people = []
    used_names: set[tuple[str, str, str]] = set()

    for person_id in range(n):
        while True:
            first = rng.choice(FIRST_NAMES)
            middle = rng.choice(MIDDLE_NAMES)
            last = rng.choice(LAST_NAMES)
            name_tuple = (first, middle, last)
            if not enforce_unique_names or name_tuple not in used_names:
                used_names.add(name_tuple)
                break

        company_name, company_city = rng.choice(COMPANIES)
        people.append({
            "id": person_id,
            "first_name": first,
            "middle_name": middle,
            "last_name": last,
            "birthmonth": rng.choice(MONTHS),
            "birthday": rng.randint(1, 28),
            "birthyear": rng.randint(1900, 2099),
            "birthcity": rng.choice(CITIES),
            "university": rng.choice(UNIVERSITIES),
            "field": rng.choice(FIELDS),
            "company1name": company_name,
            "company1city": company_city,
        })

    return people


def render_bio(
    person: dict,
    rng: random.Random,
    permute: bool = False,
    fullname: bool = False,
) -> str:
    """Render a biography from templates. Follows Bio3's get_text_simple3()."""
    full_name = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
    he_she = "He" if person["id"] % 2 == 0 else "She"
    birthday = f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}"

    s1 = rng.choice(BIRTH_DATE_TEMPLATES)
    s2 = rng.choice(BIRTH_CITY_TEMPLATES)
    s3 = rng.choice(UNIVERSITY_TEMPLATES)
    s4 = rng.choice(MAJOR_TEMPLATES)
    s5 = rng.choice(WORK_CITY_TEMPLATES)
    s6 = rng.choice(EMPLOYER_TEMPLATES)

    order = rng.randint(0, 1)
    if order == 0:
        sentences = [s1, s2, s3, s4, s5, s6]
    else:
        sentences = [s1, s2, s3, s4, s6, s5]

    if permute:
        rng.shuffle(sentences)

    rendered = []
    for i, tmpl in enumerate(sentences):
        name = full_name if (fullname or i == 0) else he_she
        rendered.append(
            " " + tmpl.format(
                name=name,
                birthday=birthday,
                birthcity=person["birthcity"],
                university=person["university"],
                field=person["field"],
                company1city=person["company1city"],
                company1name=person["company1name"],
            )
        )

    return "".join(rendered)


def render_qa(person: dict, rng: random.Random,
              templates_dict: dict[str, list[str]] | None = None) -> list[str]:
    """Render one QA pair per attribute for a person. Returns list of QA strings.

    `templates_dict` overrides the default QA_TEMPLATES (used by callers that
    want a different template-count regime, e.g. via build_qa_templates(K))."""
    full_name = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
    birthday = f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}"

    qa_pairs = []
    src = templates_dict if templates_dict is not None else QA_TEMPLATES
    for attr, templates in src.items():
        tmpl = rng.choice(templates)
        qa_text = " " + tmpl.format(
            name=full_name,
            birthday=birthday,
            birthcity=person["birthcity"],
            university=person["university"],
            field=person["field"],
            company1city=person["company1city"],
            company1name=person["company1name"],
        )
        qa_pairs.append(qa_text)
    return qa_pairs


# ---------------------------------------------------------------------------
# Binary serialization (matching existing format)
# ---------------------------------------------------------------------------


def write_data_shard(filename: str, tokens: np.ndarray):
    """Write tokens to a bin file (magic=20251013, uint32 tokens)."""
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20251013  # magic
    header[1] = 1  # version
    header[2] = len(tokens)  # ntok
    tokens_u32 = tokens.astype(np.uint32)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_u32.tobytes())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate Bio3-style synthetic biography dataset"
    )
    parser.add_argument("--n_people", type=int, default=100_000)
    parser.add_argument("--n_phrasings", type=int, default=5, help="multi-N augmentation")
    parser.add_argument("--permute", action="store_true", help="Shuffle sentence order")
    parser.add_argument("--fullname", action="store_true", help="Full name in every sentence")
    parser.add_argument("--qa", action="store_true", help="Mix in QA pairs for extractability training")
    parser.add_argument("--qa_ratio", type=float, default=0.1,
                        help="Fraction of total tokens that are QA (default: 0.1 = 10%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.01,
                        help="Fraction of people reserved for validation")
    parser.add_argument("--shard_size", type=int, default=10_000_000,
                        help="Max tokens per shard")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-named under data/)")
    parser.add_argument("--tokenizer", type=str, default="tokenizers/bpe-16384")
    parser.add_argument("--eos_token_id", type=int, default=2,
                        help="EOS token ID for separator (default: 2 = </s> in bpe16384)")
    parser.add_argument("--bare_answers", action="store_true",
                        help="Use bare-answer-only QA forms (drops the forms whose answer half "
                             "starts with '{name}' and restates the fact declaratively — those "
                             "forms exactly match the decl-probe template and create a leakage "
                             "where QA-only-trained models can complete decl probes by surface "
                             "form). When set, QA rendering uses build_qa_templates(..., bare=True) "
                             "instead of the default QA_TEMPLATES pool.")
    args = parser.parse_args()

    # Auto-name output directory
    if args.output_dir is None:
        parts = ["bio-synthetic-bpe16384"]
        parts.append(f"N{args.n_people}")
        parts.append(f"multi{args.n_phrasings}")
        if args.permute:
            parts.append("permute")
        if args.fullname:
            parts.append("fullname")
        if args.qa:
            parts.append(f"qa{int(args.qa_ratio * 100)}")
        parts.append(f"seed{args.seed}")
        args.output_dir = os.path.join("data", "-".join(parts))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    eos_id = args.eos_token_id
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"EOS token ID: {eos_id}")

    # Generate people
    print(f"Generating {args.n_people} people (seed={args.seed})...")
    people = generate_people(args.n_people, seed=args.seed)

    # Split into train/val by person (not by bio)
    n_val = max(1, int(args.n_people * args.val_fraction))
    n_train = args.n_people - n_val
    val_people = people[:n_val]
    train_people = people[n_val:]
    print(f"Train: {n_train} people, Val: {n_val} people")

    # Generate and write shards
    for split, split_people in [("val", val_people), ("train", train_people)]:
        suffix = ""
        if args.qa:
            suffix = f" + QA @ {args.qa_ratio:.0%}"
        print(f"\nGenerating {split} split ({len(split_people)} people x {args.n_phrasings} phrasings{suffix})...")
        rng = random.Random(args.seed + (0 if split == "train" else 1_000_000))

        # First pass: generate all bio tokens
        bio_sequences: list[list[int]] = []
        for person in split_people:
            for _ in range(args.n_phrasings):
                bio = render_bio(person, rng, permute=args.permute, fullname=args.fullname)
                tokens = tokenizer.encode(bio, add_special_tokens=False)
                bio_sequences.append(tokens)

        # Generate QA tokens if requested
        qa_sequences: list[list[int]] = []
        if args.qa:
            qa_rng = random.Random(args.seed + (2_000_000 if split == "train" else 3_000_000))
            # When --bare_answers is set, route through build_qa_templates(bare=True)
            # to use the non-leaky answer pool (QA_ANSWERS_BARE). Otherwise use the
            # legacy QA_TEMPLATES (8 phrasings per attribute) via the default path.
            qa_templates_dict = (
                build_qa_templates(8, bare=True) if args.bare_answers else None
            )
            for person in split_people:
                qa_texts = render_qa(person, qa_rng, templates_dict=qa_templates_dict)
                for qa_text in qa_texts:
                    tokens = tokenizer.encode(qa_text, add_special_tokens=False)
                    qa_sequences.append(tokens)

        # Compute how many QA sequences to include for the target ratio
        total_bio_tokens = sum(len(s) + 1 for s in bio_sequences)  # +1 for EOS
        if args.qa and qa_sequences:
            # qa_ratio = qa_tokens / (bio_tokens + qa_tokens)
            # => qa_tokens = bio_tokens * qa_ratio / (1 - qa_ratio)
            target_qa_tokens = int(total_bio_tokens * args.qa_ratio / (1 - args.qa_ratio))
            avg_qa_len = sum(len(s) + 1 for s in qa_sequences) / len(qa_sequences)
            n_qa_to_use = min(len(qa_sequences), max(1, int(target_qa_tokens / avg_qa_len)))
            # Shuffle and take the needed amount
            qa_rng2 = random.Random(args.seed + (4_000_000 if split == "train" else 5_000_000))
            qa_rng2.shuffle(qa_sequences)
            qa_sequences = qa_sequences[:n_qa_to_use]
            actual_qa_tokens = sum(len(s) + 1 for s in qa_sequences)
            actual_ratio = actual_qa_tokens / (total_bio_tokens + actual_qa_tokens)
            print(f"  QA: {n_qa_to_use:,} pairs, {actual_qa_tokens:,} tokens ({actual_ratio:.1%} of total)")
        else:
            qa_sequences = []

        # Interleave: distribute QA pairs evenly among bio sequences
        all_sequences = []
        if qa_sequences:
            # Insert QA pairs at regular intervals
            qa_interval = max(1, len(bio_sequences) // len(qa_sequences))
            qa_idx = 0
            for i, bio_seq in enumerate(bio_sequences):
                all_sequences.append(bio_seq)
                if qa_idx < len(qa_sequences) and (i + 1) % qa_interval == 0:
                    all_sequences.append(qa_sequences[qa_idx])
                    qa_idx += 1
            # Append any remaining QA
            while qa_idx < len(qa_sequences):
                all_sequences.append(qa_sequences[qa_idx])
                qa_idx += 1
        else:
            all_sequences = bio_sequences

        # Write shards
        shard_tokens: list[int] = []
        shard_idx = 0
        total_tokens = 0

        for seq in all_sequences:
            shard_tokens.extend(seq)
            shard_tokens.append(eos_id)

            # Write shard if full
            if len(shard_tokens) >= args.shard_size:
                fname = os.path.join(
                    args.output_dir, f"bio_{split}_{shard_idx:06d}.bin"
                )
                arr = np.array(shard_tokens[:args.shard_size], dtype=np.uint32)
                write_data_shard(fname, arr)
                total_tokens += len(arr)
                print(f"  Wrote {fname} ({len(arr):,} tokens)")
                shard_tokens = shard_tokens[args.shard_size:]
                shard_idx += 1

        # Write remaining tokens
        if shard_tokens:
            fname = os.path.join(
                args.output_dir, f"bio_{split}_{shard_idx:06d}.bin"
            )
            arr = np.array(shard_tokens, dtype=np.uint32)
            write_data_shard(fname, arr)
            total_tokens += len(arr)
            print(f"  Wrote {fname} ({len(arr):,} tokens)")

        n_total_seq = len(bio_sequences) + len(qa_sequences)
        print(f"  {split}: {n_total_seq:,} sequences, {total_tokens:,} tokens")

    # Save metadata
    import json
    meta = {
        "n_people": args.n_people,
        "n_phrasings": args.n_phrasings,
        "permute": args.permute,
        "fullname": args.fullname,
        "qa": args.qa,
        "qa_ratio": args.qa_ratio if args.qa else 0,
        "bare_answers": args.bare_answers,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "tokenizer": args.tokenizer,
        "eos_token_id": eos_id,
        "n_train_people": n_train,
        "n_val_people": n_val,
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata to {meta_path}")

    # Save people list for evaluation
    people_path = os.path.join(args.output_dir, "people.json")
    with open(people_path, "w") as f:
        json.dump(people, f)
    print(f"Saved people list to {people_path}")


if __name__ == "__main__":
    main()
