"""Parse the free-text `reason` column of the NBA injury report.

The report writes most entries as

    Injury/Illness - <Side> <Body part>; <Ailment>

but the field is typed by hand, so roughly 1,200 distinct ailment strings
and 400 distinct body-part strings appear for what is really a few dozen
clinical categories. Everything here is keyword matching with an explicit
precedence order, expressed as polars expressions so it runs over the whole
report in one pass.

Precedence matters: "Surgery - Jones Fracture" is a surgery, not a
fracture, and "ACL Repair" is a surgery, not a tear. The ordered rule lists
below encode that, most severe / most specific first.
"""

from __future__ import annotations

import polars as pl

# --------------------------------------------------------------------------
# Top-level reason category
# --------------------------------------------------------------------------
# Most health entries are written "Injury/Illness - <part>; <ailment>".
# Because that prefix literally contains the word "illness", the injury /
# illness split has to come from the payload rather than from the raw string
# — these rules only cover the *non*-health reasons, which never carry the
# prefix. First hit wins, so "Rest - Left Knee Injury Management" lands on
# rest rather than on the knee.
NON_HEALTH_RULES: list[tuple[str, str]] = [
    ("gleague", r"g league|g-league|two-way|on assignment"),
    ("suspension", r"suspension|suspended|ineligible"),
    ("rest", r"^rest\b|load management"),
    ("roster", r"trade pending|coach'?s decision|not with team|not with the team"),
    ("personal", r"personal|bereavement|family"),
    ("protocol", r"health and safety"),
    ("reconditioning", r"return to competition|reconditioning"),
]

# Bare health entries that skip the "Injury/Illness - " prefix, e.g.
# "Metacarpal Fracture", "Tendinopathy", "quad contusion".
HEALTH_KEYWORDS = (
    r"injury|sprain|strain|soreness|\bsore\b|fracture|surgery|tear|torn|"
    r"rupture|contusion|bruise|tendin|bursitis|fasciitis|spasm|impingement|"
    r"inflammation|tightness|repair|recovery|management|maintenance|procedure|"
    r"concussion|dislocat|subluxation|hyperextension|effusion|swelling|"
    r"\bpain\b|discomfort|laceration|stress reaction|avulsion|meniscectom|"
    r"illness|non-?covid|covid|\bflu\b|virus|gastro|stomach|respiratory"
)

# --------------------------------------------------------------------------
# Body region
# --------------------------------------------------------------------------
# Matched against the side-stripped body-part text, falling back to the full
# reason when the body-part slot is empty or "N/A". Ordered so that the more
# specific structure wins over the joint that contains it where that matters
# clinically (ACL before knee, achilles before ankle).
REGION_RULES: list[tuple[str, str]] = [
    ("knee_ligament", r"\bacl\b|\bpcl\b|\bmcl\b|\blcl\b|cruciate|meniscus|meniscal|patell"),
    ("achilles", r"achilles"),
    ("knee", r"\bknee\b"),
    ("ankle", r"\bankle\b|deltoid ligament|syndesmo"),
    ("foot", (r"\bfoot\b|midfoot|forefoot|navicular|metatars|plantar|\bheel\b|\btoe\b|"
             r"sesamoid|cuboid|bunion|\barch\b")),
    ("lower_leg", (r"\bcalf\b|soleus|tibia|fibula|\bshin\b|lower leg|gastrocnem|peroneal|"
                  r"\bankle tendon\b")),
    ("hamstring", r"hamstring"),
    ("quad", r"\bquad|\bthigh\b|\bfemur\b|patellar tendon"),
    ("hip_groin", (r"\bhip\b|groin|adductor|abductor|\bpelvi|\bglute|sacro|iliac|\bpsoas\b|"
                  r"hip flexor|labrum|labral")),
    ("back_core", (r"\bback\b|lumbar|thoracic|\bspine\b|\bdisc\b|oblique|abdom|\bcore\b|"
                  r"\brib\b|\bflank\b|sports hernia|hernia|\bgroin wall\b")),
    ("shoulder", (r"shoulder|clavicle|collarbone|rotator|\bac joint\b|scapula|\bdeltoid\b|"
                 r"\bbicep|\btricep|\bpec\b|pectoral")),
    ("elbow_arm", r"\belbow\b|forearm|\bucl\b|\bhumerus\b|\bulna\b|\bradius\b|\barm\b"),
    ("hand_wrist", (r"\bwrist\b|\bhand\b|\bthumb\b|finger|\bmetacarp|\bphalan|scaphoid|"
                   r"\bknuckle\b|\bnail\b")),
    ("head_face", (r"\bhead\b|\bface\b|facial|\bnasal\b|\bnose\b|concussion|\bjaw\b|\beye\b|"
                  r"\bdental\b|\btooth\b|\bteeth\b|\bmouth\b|\bear\b|\bskull\b|\borbital\b")),
    ("neck", r"\bneck\b|cervical"),
    ("systemic", (r"illness|covid|\bflu\b|virus|gastro|stomach|respiratory|migraine|headache|"
                 r"dehydrat|thrombosis|\bdvt\b|appendic|\bblood\b|cardiac|\bheart\b|"
                 r"conditioning|\bimmune\b")),
]

# --------------------------------------------------------------------------
# Ailment class (pathology)
# --------------------------------------------------------------------------
# Matched against the ailment slot, falling back to the whole reason. The
# order is the clinical severity order the report's wording supports:
# an operated or torn structure first, bone next, then soft tissue, then the
# vaguer load-management language.
AILMENT_RULES: list[tuple[str, str]] = [
    ("surgery", (r"surgery|surgical|\brepair\b|\bprocedure\b|meniscectom|reconstruct|"
                r"arthroscop|debridement|\bscope\b|fusion|implant|\bpost-?op")),
    ("rupture_tear", (r"\btear\b|\btorn\b|\btears\b|rupture|avulsion|complete tear|"
                     r"partial tear|detach")),
    ("fracture", r"fracture|\bbroken\b|stress reaction|\bbreak\b|fx\b"),
    ("concussion", r"concussion"),
    ("dislocation", r"dislocat|subluxation|instability|hyperextension|separation"),
    ("sprain", r"sprain"),
    ("strain", r"strain|\bpull\b|pulled"),
    ("tendinopathy", (r"tendin|tendon|bursitis|fasciitis|impingement|inflammation|"
                     r"irritation|synovitis|capsulitis|neuritis|neuropathy|nerve")),
    ("contusion", r"contusion|bruise|\bpointer\b|laceration|\bcut\b|\bwelt\b|hematoma"),
    ("effusion", r"effusion|swelling|\bfluid\b|\bcyst\b"),
    ("illness", (r"illness|covid|\bflu\b|virus|gastro|stomach|respiratory|\bsick\b|"
                r"thrombosis|\bdvt\b|migraine|headache")),
    ("recovery", r"recovery|rehab|reconditioning|\breturn\b"),
    ("management", r"management|maintenance|\brest\b|load|precaution|monitor"),
    ("soreness", (r"soreness|\bsore\b|tightness|\btight\b|spasm|\bpain\b|discomfort|"
                 r"stiffness|\bache\b|\bcramp")),
]


def _first_match(text: pl.Expr, rules: list[tuple[str, str]], default: str) -> pl.Expr:
    """Build a when/then chain returning the label of the first matching rule."""
    out = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.String))
    for label, pattern in rules:
        out = out.when(text.str.contains(pattern)).then(pl.lit(label))
    return out.otherwise(pl.lit(default))


def _norm(col: str = "reason") -> pl.Expr:
    """Lower-case, collapse whitespace, and blank out the placeholder values."""
    return (
        pl.col(col)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


# Some rows have the team name glued to the front of the reason, e.g.
# "Portland Trail Blazers Injury/Illness - Abdomen; Tendinopathy".
_PREFIX = r"^.*?injury/illness\s*-\s*"

_PLACEHOLDER = r"^(n/a|n/a\.|na|-|–|—|unspecified|undisclosed|)$"


def reason_features(col: str = "reason") -> list[pl.Expr]:
    """Expressions that decompose `reason` into modelling columns.

    Returns a list of named expressions, so callers can
    `df.with_columns(reason_features())`.
    """
    norm = _norm(col)

    # Strip the "Injury/Illness - " prefix (and any team name before it) to
    # isolate the "<body part>; <ailment>" payload.
    payload = norm.str.replace(_PREFIX, "")
    has_payload = norm.str.contains(r"injury/illness")

    parts = payload.str.split(";")
    body_raw = parts.list.get(0, null_on_oob=True).str.strip_chars()
    ailment_raw = (
        parts.list.slice(1).list.join("; ").str.strip_chars()
    )

    # Side lives at the front of the body-part slot.
    side = (
        pl.when(body_raw.str.contains(r"^(left|l)\b")).then(pl.lit("left"))
        .when(body_raw.str.contains(r"^(right|r)\b")).then(pl.lit("right"))
        .when(body_raw.str.contains(r"bilateral|both")).then(pl.lit("bilateral"))
        .otherwise(pl.lit("none"))
    )
    body_part = (
        body_raw.str.replace(r"^(left|right|l|r|bilateral|both)\b", "")
        .str.strip_chars()
        .str.strip_chars_start("-")
        .str.strip_chars()
    )

    # When the body-part slot is missing or a placeholder, fall back to the
    # whole reason so the region rules still get a chance.
    region_text = (
        pl.when(~has_payload | body_part.str.contains(_PLACEHOLDER))
        .then(norm)
        .otherwise(body_part + pl.lit(" ") + ailment_raw)
    )
    ailment_text = (
        pl.when(ailment_raw.str.len_chars() == 0).then(norm).otherwise(ailment_raw)
    )

    ailment_class = _first_match(ailment_text, AILMENT_RULES, "unknown")
    region = _first_match(region_text, REGION_RULES, "unknown")

    # Health entries split into injury vs illness on the payload, not on the
    # prefix. Anything the non-health rules claim keeps their label.
    non_health = _first_match(norm, NON_HEALTH_RULES, "")
    looks_health = has_payload | norm.str.contains(HEALTH_KEYWORDS)
    health_kind = (
        pl.when((region == "systemic") | (ailment_class == "illness"))
        .then(pl.lit("illness"))
        .otherwise(pl.lit("injury"))
    )
    category = (
        pl.when(non_health != "").then(non_health)
        .when(looks_health).then(health_kind)
        .otherwise(pl.lit("unknown"))
    )

    return [
        category.alias("reason_category"),
        body_part.alias("body_part_raw"),
        side.alias("body_side"),
        region.alias("body_region"),
        ailment_raw.alias("ailment_raw"),
        ailment_class.alias("ailment_class"),
        # Stage / modifier flags. These cut across the pathology class: a knee
        # can be "ACL Surgery" (surgical) or "ACL Injury Recovery" (the rehab
        # that follows), and the two have very different remaining durations.
        ailment_text.str.contains(r"surgery|surgical|repair|procedure|meniscectom|"
                                  r"reconstruct|arthroscop|post-?op").alias("is_surgical"),
        ailment_text.str.contains(r"recovery|rehab|reconditioning").alias("is_recovery_stage"),
        ailment_text.str.contains(r"management|maintenance|precaution|load").alias("is_management"),
        ailment_text.str.contains(r"stress fracture|stress reaction").alias("is_bone_stress"),
        # Mentioning one of these structures is not the same as having done
        # something catastrophic to it: "Achilles; Soreness" is a rest night.
        # The severity flag is the conjunction of structure and pathology, and
        # is built in `features`.
        norm.str.contains(r"\bacl\b|\bpcl\b|cruciate|achilles|patellar tendon")
        .alias("mentions_major_structure"),
        # A multi-part reason ("Abdominal; Strain/Left Ankle Sprain") signals a
        # player carrying more than one problem at once.
        (payload.str.count_matches(r"/") + payload.str.count_matches(r";"))
        .alias("n_reason_separators"),
        norm.str.len_chars().alias("reason_len"),
    ]


# Categories that open an injury spell. `reconditioning` is excluded as a
# *starting* reason — on its own it is the ramp-up after a long absence, not
# a new problem — but it is allowed to continue a spell (see SPELL_CONTINUE).
SPELL_START = ("injury",)

# Categories that keep an already-open spell alive. A player rehabbing an
# ACL is routinely filed as "Injury/Illness - Left Knee; Injury Recovery",
# then "Return to Competition Reconditioning", then sent to the G League on
# a rehab assignment, all without playing. Treating those as three separate
# spells would shred long absences into short ones and bias every model low.
SPELL_CONTINUE = ("injury", "reconditioning", "gleague")
