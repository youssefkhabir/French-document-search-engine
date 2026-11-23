"""Generate the synthetic test corpus in PDF, DOCX, TXT, HTML and CSV:
10 CVs (cv/), 6 job offers (offres/) and 7 documents of other kinds (divers/): a 10-page
activity report, an administrative procedure with a table, a news article, a scientific
abstract, a decree excerpt and two CSV tables.

Existing files are kept, so committed binaries do not change on every run; use --force
to rewrite everything.

All people, companies and addresses are FICTIONAL. The files contain the cases the
Phase 1 pipeline must handle:
  - a scanned (image-only) PDF                  -> must be skipped
  - multi-page PDFs with repeated header/footer -> must be removed
  - words hyphenated at line ends               -> must be rejoined
  - HTML with nav / script / style              -> must be dropped
  - one-page CVs                                -> must stay one chunk
  - a long offer (> 250 words)                  -> must be split with overlap
  - a 10-page report                            -> many chunks, a section across pages
  - a DOCX table, "Art. 1er" abbreviations      -> table text kept, no false sentence cut
They also plant retrieval pairs for later phases, e.g. a CV that says only
"ingénieur logiciel" and an offer that says "développeur".

Usage: python scripts/make_fixtures.py [--out tests/fixtures/corpus] [--force]
"""
import argparse
import csv
from pathlib import Path

import pymupdf
from docx import Document

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Content. "\n\n" separates paragraphs; "-\n" inside a word forces a line-end hyphen.
# ---------------------------------------------------------------------------

CV_LOGICIEL = """Camille Fictif
Ingénieur logiciel

Profil
Ingénieur logiciel avec cinq ans d'expérience. J'ai conçu des applications web en Python et encadré une équipe de trois personnes. Habitué aux méthodes agiles et à la revue de code.

Expérience
2022 - aujourd'hui : Ingénieur logiciel, Société Exemple Numérique, Lyon. Conception d'une plateforme de réservation en Python (Django) et PostgreSQL, mise en place de l'intégration continue, encadrement de deux stagiaires.
2020 - 2022 : Ingénieur études, Atelier Imaginaire, Grenoble. Maintenance d'API REST, écriture de tests automatisés, migration vers Docker.

Formation
2020 : Diplôme d'ingénieur en informatique, École Fictive d'Ingénieurs.

Compétences
Python, Django, FastAPI, SQL, Docker, Git, tests unitaires, architecture logicielle.

Langues
Français (langue maternelle), anglais (courant)."""

CV_ISO = """Nadia Exemple
Auditrice sécurité des systèmes d'information

Résumé
Auditrice certifiée ISO 27001 Lead Auditor depuis 2022. Huit ans d'expérience en sécurité informatique, gouvernance et gestion des risques.

Expériences professionnelles
2021 - aujourd'hui : Auditrice SSI, Cabinet Fictif Conseil, Paris. Audits de conformité ISO 27001 et ISO 27005, analyses de risques EBIOS, rédaction de politiques de sécurité, sensibilisation des équipes.
2017 - 2021 : Analyste SOC, Opérateur Imaginaire Télécom. Surveillance des incidents, gestion des vulnérabilités, réponse aux incidents.

Certifications
ISO 27001 Lead Auditor (2022), CISSP (2020).

Formation
Master Sécurité des systèmes d'information, Université Fictive, 2017."""

CV_INFIRMIERE = """Julie Modèle
Infirmière diplômée d'État

Expérience
2019 - aujourd'hui : Infirmière en service de gériatrie, Centre Hospitalier Imaginaire. Soins aux personnes âgées, préparation et administration des traitements, suivi des dossiers patients, encadrement des aides-soignants.
2016 - 2019 : Infirmière en EHPAD, Résidence Les Tilleuls Fictifs. Coordination des soins, relation avec les familles, gestion des urgences.

Formation
Diplôme d'État d'infirmier, IFSI Fictif, 2016.

Compétences
Soins techniques, gériatrie, gestion de la douleur, travail en équipe pluridisciplinaire, logiciel de dossier patient informatisé."""

CV_DATA = """Hugo Inventé
Data scientist

Profil
Spécialiste de l'apprentissage automatique et du traitement automatique du langage. Je transforme les données en modèles de prédiction utiles aux équipes métier.

Expérience
2021 - aujourd'hui : Data scientist, Assurances Fictives. Modèles de prévision de la fraude, classification de textes de sinistres, tableaux de bord pour la direction.
2019 - 2021 : Analyste de données, Banque Imaginaire. Nettoyage de données, statistiques descriptives, automatisation de rapports.

Formation
Master en statistique et apprentissage automatique, 2019.

Compétences
Python, pandas, scikit-learn, PyTorch, SQL, visualisation, TF-IDF, plongements lexicaux, word2vec."""

CV_PAIE = """Sophie Factice
Gestionnaire paie et administration du personnel

Expérience
2018 - aujourd'hui : Gestionnaire paie, Groupe Exemple Distribution. Établissement de 450 bulletins de salaire par mois, déclarations sociales nominatives, suivi des absences et des congés, réponse aux questions des salariés.
2014 - 2018 : Assistante comptable, Cabinet Comptable Imaginaire. Saisie des factures, rapprochements bancaires, préparation des bilans.

Formation
BTS Comptabilité et gestion, 2014.

Compétences
Paie, DSN, droit social, Excel avancé, logiciels de paie, rigueur et confidentialité."""

CV_COMM_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Léa Imaginaire - Chargée de communication</title>
<style>body { font-family: sans-serif; } nav { background: #333; }</style>
<script>console.log("analytics tracker"); var cookieBanner = true;</script>
</head>
<body>
<nav><a href="/">Accueil</a> | <a href="/portfolio">Portfolio</a> | <a href="/contact">Contact</a></nav>
<header><h1>Léa Imaginaire</h1><p>Chargée de communication</p></header>
<main>
<h2>Profil</h2>
<p>Chargée de communication avec six ans d'expérience dans le secteur public. J'élabore des <strong>plans de communication</strong>, je rédige des contenus et j'anime les <em>réseaux sociaux</em>.</p>
<h2>Expérience</h2>
<ul>
<li>2020 - aujourd'hui : Chargée de communication, Mairie Fictive. Rédaction du magazine municipal, relations presse, organisation d'événements.</li>
<li>2018 - 2020 : Community manager, Agence Exemple. Gestion des comptes sociaux de dix clients, création de visuels, suivi des statistiques.</li>
</ul>
<h2>Compétences</h2>
<p>Rédaction, relations presse, réseaux sociaux, PAO, gestion de projet événementiel.</p>
</main>
<footer>© 2026 Léa Imaginaire - Mentions légales - Politique de cookies</footer>
<script>document.write("tracking pixel");</script>
</body>
</html>"""

# Two-page CV with a header and footer repeated on each page, and line-end hyphens.
CV_RESEAU_PAGES = [
    """Thomas Fictif - Technicien réseaux et télécommunications

Profil
Technicien réseaux avec sept ans d'expérience en installation, configura-
tion et maintenance d'infrastructures informatiques. Habitué aux interven-
tions sur site et au support aux utilisateurs.

Expérience
2020 - aujourd'hui : Technicien réseaux, Collectivité Imaginaire. Adminis-
tration des commutateurs et des routeurs, déploiement du Wi-Fi dans vingt bâtiments publics, supervision du réseau et gestion des incidents de niveau 2.
2017 - 2020 : Technicien support, Société Exemple Services. Installation de postes de travail, téléphonie IP, assistance aux utilisateurs.""",
    """Formation
2017 : BTS Systèmes numériques, option informatique et réseaux.

Certifications
Cisco CCNA (2019), habilitation électrique.

Compétences
Réseaux locaux, VLAN, routage, pare-feu, téléphonie sur IP, supervision, câblage, documentation technique, sens du service.

Centres d'intérêt
Radioamateur, randonnée, électronique.""",
]

CV_PROF = """Marie Exemplaire
Professeure de mathématiques

Expérience
2015 - aujourd'hui : Professeure certifiée de mathématiques, Lycée Fictif. Enseignement en seconde et en terminale, préparation au baccalauréat, conception de séquences pédagogiques, accompagnement personnalisé.
2013 - 2015 : Professeure contractuelle, Collège Imaginaire.

Formation
CAPES de mathématiques, 2013. Master Mathématiques, 2012.

Compétences
Pédagogie différenciée, évaluation, outils numériques éducatifs, travail en équipe."""

CV_BTP = """Karim Inventaire
Conducteur de travaux voirie et réseaux divers

Expérience
2016 - aujourd'hui : Conducteur de travaux VRD, Entreprise Fictive de Travaux Publics. Pilotage de chantiers de voirie, assainissement et réseaux, encadrement de quinze compagnons, suivi budgétaire, relations avec les maîtres d'ouvrage publics.
2012 - 2016 : Chef de chantier, Société Exemple Bâtiment.

Formation
Licence professionnelle Travaux publics, 2012.

Compétences
Planification, métrés, lecture de plans, sécurité sur chantier, marchés publics."""

CV_SCAN = """Paul Scanné
Agent d'accueil

Expérience
Accueil physique et téléphonique, gestion du courrier, prise de rendez-vous.
Ce CV est une image : il n'a pas de couche de texte et doit être ignoré."""

OFFRE_DEV_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Offre d'emploi - Développeur backend Python (H/F)</title>
<style>.btn { color: red; }</style>
<script>window.dataLayer = window.dataLayer || [];</script>
</head>
<body>
<nav><ul><li><a href="/">Offres</a></li><li><a href="/entreprises">Entreprises</a></li><li><a href="/connexion">Connexion</a></li></ul></nav>
<main>
<h1>Développeur backend Python (H/F)</h1>
<p><b>Entreprise :</b> Startup Fictive Mobilité, Lyon. <b>Contrat :</b> CDI. <b>Télétravail :</b> deux jours par semaine.</p>
<h2>Missions</h2>
<p>Nous recherchons un développeur backend pour rejoindre notre équipe produit. Vous développerez les API de notre plateforme de covoiturage et participerez aux choix d'architecture.</p>
<ul>
<li>Développer et maintenir des services en Python</li>
<li>Concevoir des bases de données performantes</li>
<li>Participer aux revues de code et à l'amélioration continue</li>
</ul>
<h2>Profil recherché</h2>
<p>Au moins trois ans d'expérience en développement web, maîtrise de Python et de SQL. Une première expérience d'encadrement est un plus.</p>
</main>
<aside>Offres similaires : Développeur frontend, Développeur mobile</aside>
<footer>Site d'emploi imaginaire - CGU - Cookies</footer>
</body>
</html>"""

# Long, three-page offer: repeated header/footer, page numbers, hyphens, > 250 words.
OFFRE_RSSI_PAGES = [
    """Ministère Imaginaire de l'Exemple - Offre n° 2026-114

Responsable de la sécurité des systèmes d'information (RSSI)

Contexte
La direction du numérique du Ministère Imaginaire de l'Exemple pilote la transformation numérique de quarante services répartis sur tout le territoire. Elle exploite des applications critiques pour les usagers et doit garantir leur disponibilité, leur intégrité et la confidentialité des données personnelles qu'elles traitent. Dans le cadre du renforcement de sa politique de sécurité, elle recrute son ou sa responsable de la sécurité des systèmes d'information.

Missions principales
Le ou la RSSI définit et met en œuvre la politique de sécurité des systèmes d'information. Il ou elle conduit l'analyse des risques selon la méthode EBIOS, propose les mesures de traite-
ment adaptées et suit leur mise en œuvre avec les équipes techniques. Il ou elle prépare la certification ISO 27001 du centre de données et coordonne les audits internes et externes.

Page 1 / 3""",
    """Ministère Imaginaire de l'Exemple - Offre n° 2026-114

Le ou la RSSI pilote la gestion des incidents de sécurité, en lien avec le centre opérationnel de sécurité, et organise les exercices de crise. Il ou elle veille à la conformité réglementaire, notamment au règlement général sur la protection des données, en collaboration avec le délégué à la protection des données.

Il ou elle conçoit et anime des actions de sensibilisation destinées à l'ensemble des agents : formations en ligne, campagnes de prévention de l'hameçonnage, guides de bonnes pratiques. Il ou elle conseille la direction sur les choix d'architecture et intègre la sécurité dès la conception des nouveaux projets numériques.

Profil recherché
Formation supérieure en informatique ou en sécurité des systèmes d'information. Expérience confirmée d'au moins cinq ans dans la gestion des risques, l'audit ou la gouvernance de la sécurité. Une certification ISO 27001 Lead Auditor ou Lead Implementer serait fortement appréciée.

Page 2 / 3""",
    """Ministère Imaginaire de l'Exemple - Offre n° 2026-114

Compétences attendues
Connaissance des normes ISO 27001 et ISO 27005, maîtrise des méthodes d'analyse de risques, capacité à vulgariser des sujets techniques auprès de publics variés, sens de la négociation et aptitude au manage-
ment transversal.

Conditions
Poste de catégorie A ouvert aux titulaires et aux contractuels, basé à Paris. Télétravail possible deux jours par semaine. Candidature : lettre de motivation et curriculum vitae.

Page 3 / 3""",
]

OFFRE_INFIRMIER = """Offre d'emploi : Infirmier ou infirmière en EHPAD (H/F)

Établissement : Résidence Fictive du Parc, établissement d'hébergement pour personnes âgées dépendantes.
Contrat : CDI à temps plein, travail un week-end sur deux.

Missions
Vous assurez les soins techniques et relationnels auprès des résidents, la préparation et la distribution des médicaments, la coordination avec le médecin traitant et l'accompagnement des familles. Vous encadrez l'équipe d'aides-soignants.

Profil
Diplôme d'État d'infirmier exigé. Une expérience auprès des personnes âgées est appréciée. Bienveillance, patience et sens de l'organisation."""

OFFRE_PAIE = """Offre d'emploi : Gestionnaire de paie (H/F)

Collectivité Exemple recrute un ou une gestionnaire de paie pour son service des ressources humaines.

Missions
Préparer et contrôler les bulletins de salaire des agents, réaliser les déclarations sociales, gérer les dossiers de retraite et les arrêts maladie, répondre aux questions des agents sur leur rémunération.

Profil
Formation en comptabilité ou en ressources humaines. Maîtrise d'un logiciel de paie et d'Excel. Discrétion professionnelle exigée."""

OFFRE_DATA = """Offre d'emploi : Data analyst (H/F)

Mutuelle Imaginaire recherche un ou une data analyst pour sa direction de la stratégie.

Missions
Collecter et nettoyer les données de nos adhérents, construire des tableaux de bord, réaliser des études statistiques et présenter les résultats aux équipes métier.

Profil
Formation supérieure en statistique ou en informatique. Maîtrise de SQL et de Python, connaissance d'un outil de visualisation. Un intérêt pour l'apprentissage automatique est un plus."""

OFFRE_PROF_HTML = """<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><title>Recrutement de professeurs contractuels en mathématiques</title>
<script>trackPage();</script></head>
<body>
<nav>Académie Fictive | Recrutement | Nous contacter</nav>
<h1>Recrutement de professeurs contractuels en mathématiques</h1>
<p>L'Académie Fictive recrute des professeurs contractuels de mathématiques pour des remplacements dans les collèges et les lycées de la région.</p>
<p>Vous enseignerez les mathématiques à des classes de collège ou de lycée, préparerez vos cours et évaluerez les élèves en lien avec l'équipe pédagogique.</p>
<p>Profil : licence ou master de mathématiques, goût pour la transmission et le travail avec les adolescents.</p>
<footer>Académie Fictive - Mentions légales</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = 595, 842  # A4 in points
MARGIN = 60


FORCE = False  # set by --force


def keep_existing(writer):
    """Skip writing when the file exists, unless --force: committed binaries stay stable."""
    def wrapped(path: Path, *args, **kwargs):
        if path.exists() and not FORCE:
            return
        writer(path, *args, **kwargs)
    wrapped.__name__ = writer.__name__
    return wrapped


@keep_existing
def write_pdf(path: Path, pages: list[str], title: str, fontsize: float = 10.5) -> None:
    """One string per page. Lines are placed as written, so '-\\n' gives a real line-end hyphen."""
    doc = pymupdf.open()
    for text in pages:
        # Built-in Helvetica has no glyph for the oe ligature; real documents keep it.
        text = text.replace("œ", "oe")
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        rect = pymupdf.Rect(MARGIN, MARGIN, PAGE_W - MARGIN, PAGE_H - MARGIN)
        rc = page.insert_textbox(rect, text, fontsize=fontsize, fontname="helv")
        if rc < 0:
            raise ValueError(f"text overflows the page in {path.name}")
    doc.set_metadata({"title": title, "author": "Fictitious test data"})
    doc.save(path)


@keep_existing
def write_scanned_pdf(path: Path, text: str) -> None:
    """Render a text page to an image and put only the image in the PDF (no text layer)."""
    src = pymupdf.open()
    page = src.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_textbox(pymupdf.Rect(MARGIN, MARGIN, PAGE_W - MARGIN, PAGE_H - MARGIN), text,
                        fontsize=12, fontname="helv")
    pix = page.get_pixmap(dpi=80, colorspace=pymupdf.csGRAY)
    out = pymupdf.open()
    p = out.new_page(width=PAGE_W, height=PAGE_H)
    p.insert_image(p.rect, stream=pix.tobytes("jpeg", jpg_quality=60))  # keep the file small
    out.save(path)


@keep_existing
def write_docx(path: Path, text: str, title: str, header: str | None = None) -> None:
    doc = Document()
    doc.core_properties.title = title
    if header:
        doc.sections[0].header.paragraphs[0].text = header
    for para in text.split("\n\n"):
        lines = para.split("\n")
        if len(lines) == 1 or len(lines[0]) > 40:
            doc.add_paragraph(para.replace("\n", " "))
        else:  # first short line is a section heading
            doc.add_heading(lines[0], level=2)
            for line in lines[1:]:
                doc.add_paragraph(line)
    doc.save(path)


@keep_existing
def write_txt(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.write_text(text, encoding=encoding)


# Training catalogue, French Excel style: ";" delimiter, Windows-1252, a description with a
# line break inside a quoted cell, and one empty cell (no "Prérequis" for the last course).
CSV_FORMATIONS = [
    ["Intitulé", "Domaine", "Durée", "Lieu", "Description", "Prérequis"],
    ["Initiation à la cybersécurité", "Informatique", "2 jours", "Lyon",
     "Comprendre les menaces courantes : hameçonnage, rançongiciels, mots de passe faibles.\n"
     "Adopter les bons réflexes au quotidien.", "Aucun"],
    ["Gestion de projet agile", "Management", "3 jours", "Paris",
     "Découvrir Scrum et Kanban, animer les cérémonies, prioriser un carnet de produit.",
     "Expérience en conduite de projet"],
    ["Premiers secours en milieu professionnel", "Santé et sécurité", "1 jour", "Grenoble",
     "Réagir face à un malaise, pratiquer le massage cardiaque et utiliser un défibrillateur.",
     "Aucun"],
    ["Rédiger des comptes rendus efficaces", "Communication", "1 jour", "À distance",
     "Structurer un compte rendu de réunion, aller à l'essentiel, adapter le ton au lecteur.",
     ""],
]

# Numeric table, "," delimiter, UTF-8: little text besides the column names and labels.
CSV_INDICATEURS = [
    ["Région", "Année", "Taux de chômage (%)", "Population active"],
    ["Auvergne-Rhône-Alpes", "2025", "6.4", "4100000"],
    ["Île-de-France", "2025", "7.1", "6300000"],
    ["Occitanie", "2025", "8.6", "2700000"],
]


@keep_existing
def write_csv(path: Path, rows: list[list[str]], delimiter: str, encoding: str) -> None:
    with path.open("w", encoding=encoding, newline="") as f:
        csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL).writerows(rows)


# ---------------------------------------------------------------------------
# divers/: documents that are not CVs or offers
# ---------------------------------------------------------------------------

# Ten pages, one section per page, with a header and "Page n / 10" footer on each.
# Section 6 continues on page 7 (no heading there), to test a chunk that spans pages.
REPORT_TITLE = "Agence Fictive de l'Eau et de l'Environnement - Rapport d'activité 2025"
REPORT_SECTIONS = [
    """1. Mot de la directrice
L'année 2025 a été marquée par une sécheresse précoce et par la mise en place de notre nouveau plan stratégique. Malgré un contexte budgétaire tendu, l'agence a maintenu l'ensemble de ses missions de surveillance et a renforcé son action auprès des collectivités. Je remercie les équipes pour leur engagement constant. Ce rapport présente nos résultats dans six domaines : la qualité de l'eau, la biodiversité, la qualité de l'air, les déchets, l'énergie et la sensibilisation du public. Il détaille aussi nos moyens humains et financiers, ainsi que nos priorités pour l'année 2026.""",
    """2. Qualité de l'eau
Le réseau de surveillance compte désormais 412 stations de mesure réparties sur les cours d'eau et les nappes souterraines du bassin. En 2025, 78 % des masses d'eau superficielles ont atteint le bon état écologique, contre 74 % en 2024. Les concentrations en nitrates restent élevées dans les zones de grandes cultures, où un programme d'accompagnement des agriculteurs a été lancé. Les épisodes de faibles débits estivaux ont conduit à des arrêtés de restriction d'usage dans onze départements. L'agence a financé la modernisation de vingt-trois stations d'épuration, pour un montant total de 14 millions d'euros.""",
    """3. Biodiversité
Les inventaires naturalistes ont recensé 2 140 espèces sur les sites suivis par l'agence, dont 96 espèces protégées. La restauration de la continuité écologique a progressé avec l'effacement de dix-sept seuils sur les rivières, ce qui permet de nouveau la remontée des poissons migrateurs comme le saumon et l'anguille. Trois zones humides ont été acquises et seront gérées en partenariat avec les conservatoires d'espaces naturels. Un suivi des pollinisateurs a été conduit avec des bénévoles dans quarante communes.""",
    """4. Qualité de l'air
Les capteurs de l'agence ont enregistré une baisse de 9 % des concentrations moyennes de dioxyde d'azote dans les agglomérations, liée au renouvellement du parc de véhicules et aux zones à faibles émissions. En revanche, les épisodes de pollution à l'ozone ont été plus nombreux pendant l'été, en raison des fortes chaleurs. Quatorze alertes ont été diffusées à la population. Une campagne de mesure des pesticides dans l'air a été menée à proximité des écoles situées en zone agricole.""",
    """5. Gestion des déchets
Le taux de valorisation des déchets ménagers a atteint 61 % sur le territoire. L'agence a soutenu la création de huit ressourceries et l'extension du tri des biodéchets à 310 000 habitants supplémentaires. Les dépôts sauvages restent un problème majeur dans les zones rurales : un dispositif de signalement en ligne a permis de traiter 1 250 signalements. Une étude sur le gaspillage alimentaire dans la restauration collective a été publiée et sert de base à des actions dans les cantines scolaires.""",
    """6. Énergie et climat
Le bilan des émissions de gaz à effet de serre du territoire montre une baisse de 3 % par rapport à 2024. L'agence a accompagné quarante projets de production d'énergie renouvelable, principalement des installations photovoltaïques sur des bâtiments publics et des réseaux de chaleur alimentés par la biomasse. Le programme de rénovation énergétique des logements a aidé 2 900 ménages modestes à isoler leur habitation et à remplacer leur chaudière. L'adaptation au changement climatique devient une priorité : les collectivités demandent un appui pour végétaliser les cours d'école et pour gérer les eaux pluviales lors des orages violents.""",
    """Les diagnostics de vulnérabilité réalisés dans douze intercommunalités ont identifié les quartiers les plus exposés aux vagues de chaleur. Ces résultats orientent désormais les financements de l'agence vers les projets de désimperméabilisation des sols et de création d'îlots de fraîcheur. Une première évaluation montre une baisse de plusieurs degrés de la température ressentie dans les espaces réaménagés.""",
    """7. Sensibilisation du public
Plus de 18 000 élèves ont participé aux animations pédagogiques de l'agence sur l'eau, les déchets et la biodiversité. Le site internet a reçu 640 000 visites et les fiches pratiques sur les économies d'eau ont été les plus consultées. Une exposition itinérante sur le changement climatique a circulé dans vingt médiathèques. L'agence a également formé 350 agents des collectivités aux enjeux de la transition écologique.""",
    """8. Ressources humaines
Au 31 décembre 2025, l'agence employait 286 agents, dont 54 % de femmes. Vingt et un recrutements ont été réalisés, principalement des ingénieurs et des techniciens de terrain. Le plan de formation a représenté 4 200 heures, avec un effort particulier sur la sécurité lors des prélèvements en rivière. Le télétravail concerne désormais 70 % des agents dont les missions le permettent. Un baromètre social montre une satisfaction globale en hausse.""",
    """9. Budget
Le budget exécuté s'élève à 212 millions d'euros. Les aides versées aux collectivités, aux entreprises et aux agriculteurs représentent 71 % des dépenses. Les recettes proviennent pour l'essentiel des redevances sur l'eau, complétées par des subventions européennes. Les dépenses de fonctionnement ont été contenues à 9 % du budget grâce à la mutualisation des moyens informatiques et à la réduction des surfaces de bureaux.""",
    """10. Perspectives 2026
En 2026, l'agence concentrera ses moyens sur trois priorités : la sobriété dans les usages de l'eau, la restauration des milieux naturels et l'adaptation des territoires au changement climatique. Un nouvel appel à projets sera lancé pour les solutions fondées sur la nature. La refonte du réseau de mesure de la qualité de l'air sera achevée, et un observatoire des données environnementales ouvrira au public.""",
]

PROCEDURE_INTRO = """Procédure : demande de carte nationale d'identité

La carte nationale d'identité est gratuite lors d'une première demande ou d'un renouvellement. Elle est payante en cas de perte ou de vol. La demande se fait en mairie, sur rendez-vous, dans n'importe quelle commune équipée d'un dispositif de recueil.

Étapes
Remplir une pré-demande en ligne et noter le numéro obtenu. Prendre rendez-vous auprès d'une mairie équipée. Se présenter en personne avec les pièces justificatives : les empreintes digitales sont relevées sur place. Retirer la carte dans la même mairie lorsque le SMS de mise à disposition est reçu."""
PROCEDURE_TABLE = [
    ["Situation", "Pièces à fournir", "Coût"],
    ["Première demande", "Photo d'identité récente, justificatif de domicile, acte de naissance si nécessaire", "Gratuit"],
    ["Renouvellement", "Ancienne carte, photo d'identité, justificatif de domicile", "Gratuit"],
    ["Perte ou vol", "Déclaration de perte ou de vol, photo d'identité, justificatif de domicile, timbre fiscal", "25 euros"],
]
PROCEDURE_END = "Le délai de fabrication varie de deux à six semaines selon la période de l'année."

ARTICLE_HTML = """<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><title>Le vélo gagne du terrain dans les villes moyennes - Journal Imaginaire</title>
<script>loadAds();</script></head>
<body>
<nav><a href="/">Une</a> | <a href="/societe">Société</a> | <a href="/sport">Sport</a></nav>
<article>
<h1>Le vélo gagne du terrain dans les villes moyennes</h1>
<p class="chapo">Pistes cyclables, aides à l'achat, stationnements sécurisés : les villes de taille moyenne rattrapent leur retard sur les métropoles.</p>
<p>À Villefictive, la part des trajets domicile-travail effectués à <strong>vélo</strong> a doublé en trois ans. La municipalité a créé quarante kilomètres d'aménagements cyclables et installé des abris fermés près de la gare.</p>
<p>« Le plus difficile était de convaincre les commerçants », explique l'adjointe chargée des mobilités. Les enquêtes montrent pourtant que les clients venus à pied ou à vélo dépensent davantage sur un mois que ceux venus en voiture.</p>
<p>Les associations d'usagers demandent maintenant des itinéraires continus vers les communes voisines, où les routes restent dangereuses pour les cyclistes.</p>
</article>
<aside>À lire aussi : Les transports en commun gratuits, une fausse bonne idée ?</aside>
<footer>Journal Imaginaire - Tous droits réservés</footer>
</body>
</html>"""

RESUME_SCIENTIFIQUE = """Effet de la végétalisation des cours d'école sur la température ressentie : une étude dans cinq villes françaises

Résumé
Les cours d'école minérales deviennent des îlots de chaleur lors des épisodes caniculaires. Cette étude compare dix cours réaménagées avec des arbres, des sols perméables et des points d'eau à dix cours restées en bitume, dans cinq villes. Des capteurs ont mesuré la température de l'air et la température de surface pendant deux étés. L'indice de confort thermique a été calculé chaque heure.

Résultats
Dans les cours végétalisées, la température de surface est inférieure de 12 °C en moyenne en début d'après-midi, et la température ressentie baisse de 3 à 5 °C. L'effet est maximal sous les arbres adultes et faible sous les jeunes plantations.

Conclusion
La désimperméabilisation des cours d'école est une mesure efficace d'adaptation au changement climatique. Son bénéfice augmente avec la croissance des arbres, ce qui plaide pour des plantations rapides.

Mots-clés : îlot de chaleur urbain, adaptation, confort thermique, école, végétalisation."""

DECRET_PAGES = ["""Décret fictif n° 2026-41 du 12 février 2026 relatif aux horaires d'ouverture des déchetteries intercommunales

Le Premier ministre fictif,
Vu le code de l'environnement, notamment son art. L. 541-1 ;
Vu l'avis de la commission consultative d'évaluation des normes ;

Décrète :

Art. 1er. - Les déchetteries gérées par un établissement public de coopération intercommunale sont ouvertes au moins cinq jours par semaine, dont le samedi. Les horaires sont affichés à l'entrée du site et publiés en ligne.

Art. 2. - Pendant la période estivale, définie du 1er juin au 31 août, l'ouverture peut être avancée à sept heures afin d'éviter les fortes chaleurs. Les dispositions de l'art. 1er restent applicables.

Art. 3. - Le présent décret entre en vigueur le 1er avril 2026. M. le ministre fictif de la transition écologique est chargé de son exécution."""]


@keep_existing
def write_docx_table(path: Path, intro: str, table: list[list[str]], end: str, title: str) -> None:
    doc = Document()
    doc.core_properties.title = title
    for para in intro.split("\n\n"):
        doc.add_paragraph(para.replace("\n", " "))
    t = doc.add_table(rows=len(table), cols=len(table[0]))
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            t.cell(i, j).text = cell
    doc.add_paragraph(end)
    doc.save(path)


def main() -> None:
    global FORCE
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "tests" / "fixtures" / "corpus"))
    p.add_argument("--force", action="store_true", help="rewrite files that already exist")
    args = p.parse_args()
    FORCE = args.force
    out = Path(args.out)
    (out / "cv").mkdir(parents=True, exist_ok=True)
    (out / "offres").mkdir(parents=True, exist_ok=True)

    write_pdf(out / "cv" / "cv_ingenieur_logiciel.pdf", [CV_LOGICIEL], "CV - Ingénieur logiciel")
    write_docx(out / "cv" / "cv_auditrice_iso27001.docx", CV_ISO, "CV - Auditrice sécurité ISO 27001",
               header="Curriculum vitae - document confidentiel")
    write_txt(out / "cv" / "cv_infirmiere.txt", CV_INFIRMIERE)
    write_pdf(out / "cv" / "cv_data_scientist.pdf", [CV_DATA], "CV - Data scientist")
    write_docx(out / "cv" / "cv_gestionnaire_paie.docx", CV_PAIE, "CV - Gestionnaire paie")
    write_txt(out / "cv" / "cv_chargee_communication.html", CV_COMM_HTML)
    header = "Thomas Fictif - CV - "
    write_pdf(out / "cv" / "cv_technicien_reseaux.pdf",
              [f"{header}page {i + 1}\n\n{t}\n\nthomas.fictif@example.org - {i + 1}"
               for i, t in enumerate(CV_RESEAU_PAGES)],
              "CV - Technicien réseaux")
    # Windows-1252 on purpose: the loader must not assume UTF-8.
    write_txt(out / "cv" / "cv_professeure_maths.txt", CV_PROF, encoding="cp1252")
    write_docx(out / "cv" / "cv_conducteur_travaux.docx", CV_BTP, "CV - Conducteur de travaux VRD")
    write_scanned_pdf(out / "cv" / "cv_scanne_agent_accueil.pdf", CV_SCAN)

    write_txt(out / "offres" / "offre_developpeur_backend.html", OFFRE_DEV_HTML)
    write_pdf(out / "offres" / "offre_rssi_ministere.pdf", OFFRE_RSSI_PAGES,
              "Offre - Responsable de la sécurité des systèmes d'information")
    write_txt(out / "offres" / "offre_infirmier_ehpad.txt", OFFRE_INFIRMIER)
    write_docx(out / "offres" / "offre_gestionnaire_paie.docx", OFFRE_PAIE, "Offre - Gestionnaire de paie")
    write_pdf(out / "offres" / "offre_data_analyst.pdf", [OFFRE_DATA], "Offre - Data analyst")
    write_txt(out / "offres" / "offre_professeur_maths.html", OFFRE_PROF_HTML)

    (out / "divers").mkdir(parents=True, exist_ok=True)
    write_csv(out / "divers" / "catalogue_formations.csv", CSV_FORMATIONS, ";", "cp1252")
    write_csv(out / "divers" / "indicateurs_emploi.csv", CSV_INDICATEURS, ",", "utf-8")

    n = len(REPORT_SECTIONS)
    write_pdf(out / "divers" / "rapport_activite_2025.pdf",
              [f"{REPORT_TITLE}\n\n{s}\n\nPage {i + 1} / {n}" for i, s in enumerate(REPORT_SECTIONS)],
              REPORT_TITLE)
    write_docx_table(out / "divers" / "procedure_carte_identite.docx", PROCEDURE_INTRO,
                     PROCEDURE_TABLE, PROCEDURE_END, "Procédure - Carte nationale d'identité")
    write_txt(out / "divers" / "article_velo_villes_moyennes.html", ARTICLE_HTML)
    write_txt(out / "divers" / "resume_vegetalisation_ecoles.txt", RESUME_SCIENTIFIQUE)
    # Title left empty on purpose: the loader must fall back to the file name.
    write_pdf(out / "divers" / "decret_horaires_dechetteries.pdf", DECRET_PAGES, "Microsoft Word - decret_v3.docx")

    files = sorted(f for f in out.rglob("*") if f.is_file())
    print(f"{len(files)} files in {out}")
    for f in files:
        print(f"  {f.relative_to(out)}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
