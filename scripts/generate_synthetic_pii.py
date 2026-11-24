import json
import random
import re
from pathlib import Path
from faker import Faker

fake = Faker("ru_RU")
Faker.seed(42)
random.seed(42)

# Соответствие тегов стандарту wikiann/ru (7 классов)
LABEL2ID = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
}

TEMPLATES = [
    "Договор №{doc_num} заключен между {org1} и {org2} в лице представителя {person}. Паспорт {passport}, ИНН {inn}, телефон {phone}.",
    "Клиент {person} из города {city} (email: {email}, тел. {phone}) запросил выписку в банке {org1} по карте {card}.",
    "Заявление от сотрудника компании {org1}: {person}, СНИЛС {snils}, паспорт {passport}. Документ направлен в ФНС России.",
    "Акт выполненных работ подписан подрядчиком {org1} ({person}) и заказчиком {org2}. Реквизиты: ИНН {inn}, карта {card}.",
    "В Арбитражный суд города {city} поступило обращение от {person} ({org1}, тел. {phone}, паспорт {passport}).",
]


def generate_company() -> str:
    forms = ["ООО", "ПАО", "АО", "ЗАО"]
    names = ["Вектор", "Альфа-Логистик", "Северсталь-Проект", "ТехноСтрой", "Газпром-Медиа", "Сириус", "Орион", "Балтика"]
    return f"{random.choice(forms)} «{random.choice(names)}»"


def generate_passport() -> str:
    return f"{random.randint(1000, 9999)} {random.randint(100000, 999999)}"


def generate_inn() -> str:
    return "".join([str(random.randint(0, 9)) for _ in range(12)])


def generate_snils() -> str:
    return (
        f"{random.randint(100, 999)}-"
        f"{random.randint(100, 999)}-"
        f"{random.randint(100, 999)} "
        f"{random.randint(10, 99)}"
    )


def generate_card() -> str:
    return " ".join([str(random.randint(1000, 9999)) for _ in range(4)])


def generate_phone() -> str:
    return (
        f"+7 ({random.randint(900, 999)}) "
        f"{random.randint(100, 999)}-"
        f"{random.randint(10, 99)}-"
        f"{random.randint(10, 99)}"
    )


def build_bio_tokens(text: str, spans: list[tuple[str, str]]) -> tuple[list[str], list[int]]:
    """Разбивает текст на токены и проставляет BIO-метки для PER, ORG, LOC."""
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    tags = ["O"] * len(tokens)

    for entity_text, label_prefix in spans:
        ent_tokens = re.findall(r"\w+|[^\w\s]", entity_text, re.UNICODE)
        if not ent_tokens:
            continue
        n = len(ent_tokens)
        for i in range(len(tokens) - n + 1):
            if tokens[i : i + n] == ent_tokens and all(t == "O" for t in tags[i : i + n]):
                tags[i] = f"B-{label_prefix}"
                for j in range(1, n):
                    tags[i + j] = f"I-{label_prefix}"
                break

    tag_ids = [LABEL2ID[t] for t in tags]
    return tokens, tag_ids


def main(num_samples: int = 300, output_path: str = "data/raw/synthetic_pii.jsonl") -> None:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(num_samples):
        person = fake.name()
        org1 = generate_company()
        org2 = generate_company()
        while org2 == org1:
            org2 = generate_company()
        city = fake.city_name()

        regex_entities = {
            "PASSPORT_RU": generate_passport(),
            "INN": generate_inn(),
            "SNILS": generate_snils(),
            "CARD": generate_card(),
            "PHONE_RU": generate_phone(),
            "EMAIL": fake.email(),
        }

        template = random.choice(TEMPLATES)
        text = template.format(
            doc_num=random.randint(100, 9999),
            person=person,
            org1=org1,
            org2=org2,
            city=city,
            passport=regex_entities["PASSPORT_RU"],
            inn=regex_entities["INN"],
            snils=regex_entities["SNILS"],
            card=regex_entities["CARD"],
            phone=regex_entities["PHONE_RU"],
            email=regex_entities["EMAIL"],
        )

        ner_spans = [(person, "PER"), (org1, "ORG"), (org2, "ORG"), (city, "LOC")]
        tokens, ner_tags = build_bio_tokens(text, ner_spans)

        present_regex = {k: v for k, v in regex_entities.items() if v in text}
        present_ner = {
            "PER": [person] if person in text else [],
            "ORG": [o for o in [org1, org2] if o in text],
            "LOC": [city] if city in text else [],
        }

        records.append({
            "id": i + 1,
            "text": text,
            "tokens": tokens,
            "ner_tags": ner_tags,
            "ner_entities": present_ner,
            "regex_entities": present_regex,
        })

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Сгенерировано {len(records)} примеров с BIO-разметкой в {out_file}")


if __name__ == "__main__":
    main()
