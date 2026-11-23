"""Generate the synthetic test corpus: 10 CVs + 6 job offers in PDF, DOCX, TXT and HTML.

All people, companies and addresses are FICTIONAL. The files contain the cases the
Phase 1 pipeline must handle:
  - a scanned (image-only) PDF                  -> must be skipped
  - multi-page PDFs with repeated header/footer -> must be removed
  - words hyphenated at line ends               -> must be rejoined
  - HTML with nav / script / style              -> must be dropped
  - one-page CVs                                -> must stay one chunk
  - a long offer (> 250 words)                  -> must be split with overlap
They also plant retrieval pairs for later phases, e.g. a CV that says only
"ingénieur logiciel" and an offer that says "développeur".

Usage: python scripts/make_fixtures.py [--out tests/fixtures/corpus]
"""
import argparse
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


def write_txt(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.write_text(text, encoding=encoding)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "tests" / "fixtures" / "corpus"))
    out = Path(p.parse_args().out)
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

    files = sorted(f for f in out.rglob("*") if f.is_file())
    print(f"{len(files)} files written to {out}")
    for f in files:
        print(f"  {f.relative_to(out)}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
