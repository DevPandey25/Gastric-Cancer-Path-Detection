"""
Authoritative Medical Evidence and Clinical Guideline Source Registry.
GastricPath AI — Phase 1: Medical Evidence Source Audit.

Provides a maintained registry of verified clinical guidelines, pathology reporting standards,
biomarker companion diagnostics, and research reporting standards with jurisdiction support.
Literature cutoff: November 12, 2026.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Optional


class EvidenceLevel(str, Enum):
    GUIDELINE = "1A: Clinical Practice Guideline (ESMO / ASCO / NCCN)"
    REGULATORY = "1B: Government / Regulatory Label (FDA / EMA / CDSCO)"
    PATHOLOGY_STANDARD = "1C: Consensus Pathology Standard (WHO / CAP / ICCR)"
    RCT = "2A: Randomized Controlled Trial"
    META_ANALYSIS = "2B: Systematic Review & Meta-Analysis"
    COHORT_STUDY = "3: Peer-Reviewed Cohort / Observational Study"
    PREPRINT = "4: Non-Peer-Reviewed Preprint (Explicitly Disclosed)"


class Jurisdiction(str, Enum):
    INDIA = "India"
    US = "United States"
    EU = "European Union"
    JAPAN = "Japan"
    SOUTH_KOREA = "South Korea"
    GLOBAL = "Global / General"


@dataclass
class EvidenceSource:
    source_id: str
    title: str
    organization: str
    publication_date: str
    version_or_edition: str
    url: str
    evidence_level: EvidenceLevel
    jurisdiction: Jurisdiction
    summary: str
    key_recommendations: list[str] = field(default_factory=list)
    doi_or_pmid: Optional[str] = None
    access_date: str = "2026-11-12"
    verification_status: str = "Verified Current"


# ---------------------------------------------------------------------------
# Authoritative Guidelines Registry
# ---------------------------------------------------------------------------
GUIDELINE_REGISTRY: dict[str, EvidenceSource] = {
    "ESMO_2026": EvidenceSource(
        source_id="ESMO_2026",
        title="ESMO Clinical Practice Guideline for the Diagnosis, Treatment and Follow-Up of Gastric and Oesophageal Cancer",
        organization="European Society for Medical Oncology (ESMO)",
        publication_date="2026-10-15",
        version_or_edition="2026 Update (Annals of Oncology)",
        url="https://www.esmo.org/guidelines/esmo-clinical-practice-guideline-gastric-and-oesophageal-cancer",
        evidence_level=EvidenceLevel.GUIDELINE,
        jurisdiction=Jurisdiction.EU,
        doi_or_pmid="10.1016/j.annonc.2026.08.002",
        summary="Definitive clinical guidelines covering staging, multimodal perioperative therapy, and systemic biomarker-directed therapies for gastric adenocarcinoma.",
        key_recommendations=[
            "Perioperative FLOT chemotherapy recommended for resectable stage >= cT2 and/or cN+ gastric adenocarcinoma.",
            "Mandatory reflex testing for HER2, PD-L1 (CPS), mismatch repair (MMR/MSI), and CLDN18.2 in advanced/metastatic disease.",
            "Zolbetuximab plus mFOLFOX6 recommended for CLDN18.2-positive (>=75% cells with >=2+ intensity), HER2-negative metastatic disease.",
            "Nivolumab plus fluoropyrimidine/platinum recommended in HER2-negative, PD-L1 CPS >= 5 first-line metastatic adenocarcinoma.",
            "Trastuzumab plus chemotherapy recommended for HER2-positive (IHC 3+ or IHC 2+/ISH+) first-line metastatic adenocarcinoma."
        ]
    ),
    "ESMO_LIVING_2026": EvidenceSource(
        source_id="ESMO_LIVING_2026",
        title="ESMO Living Guideline: Gastric and Oesophageal Cancer",
        organization="European Society for Medical Oncology (ESMO)",
        publication_date="2026-11-01",
        version_or_edition="Living Update Q4 2026",
        url="https://www.esmo.org/guidelines/living-guidelines/esmo-living-guideline-gastric-and-oesophageal-cancer",
        evidence_level=EvidenceLevel.GUIDELINE,
        jurisdiction=Jurisdiction.GLOBAL,
        summary="Dynamically updated clinical recommendations incorporating newly reported clinical trials in gastroesophageal immunotherapy.",
        key_recommendations=[
            "Continuous review of emerging antibody-drug conjugates (ADCs) in HER2-low and Claudin-18.2 refractory disease.",
            "Stratification of adjuvant immunotherapy in dMMR/MSI-H resected disease."
        ]
    ),
    "ASCO_2026": EvidenceSource(
        source_id="ASCO_2026",
        title="Immunotherapy and Targeted Therapy for Advanced Gastroesophageal Cancer: ASCO Guideline Update",
        organization="American Society of Clinical Oncology (ASCO)",
        publication_date="2026-06-15",
        version_or_edition="2026 Rapid Update",
        url="https://ascopubs.org/guidelines/immunotherapy-and-targeted-therapy-advanced-gastroesophageal-cancer",
        evidence_level=EvidenceLevel.GUIDELINE,
        jurisdiction=Jurisdiction.US,
        doi_or_pmid="10.1200/JCO.2026.44.1102",
        summary="Evidence-based guidelines on checkpoint inhibitors and targeted agents in advanced HER2+, PD-L1+, MSI-H, and CLDN18.2+ gastroesophageal carcinoma.",
        key_recommendations=[
            "First-line pembrolizumab or nivolumab combined with platinum/fluoropyrimidine chemotherapy for patients with PD-L1 CPS >= 10 (or CPS >= 5 for nivolumab).",
            "Trastuzumab deruxtecan (T-DXd) recommended in second- or third-line for HER2-positive metastatic disease progressing on trastuzumab.",
            "Zolbetuximab plus chemotherapy for CLDN18.2-positive advanced disease (SPOTLIGHT and GLOW criteria)."
        ]
    ),
    "NCI_PDQ_2026": EvidenceSource(
        source_id="NCI_PDQ_2026",
        title="Gastric Cancer Treatment (PDQ) — Health Professional Version",
        organization="National Cancer Institute (NCI) / National Institutes of Health (NIH)",
        publication_date="2026-09-01",
        version_or_edition="PDQ 2026 Autumn Revision",
        url="https://www.cancer.gov/types/stomach/hp/stomach-treatment-pdq",
        evidence_level=EvidenceLevel.REGULATORY,
        jurisdiction=Jurisdiction.US,
        summary="Comprehensive US government health professional summary of gastric cancer staging, surgery, systemic therapy, and supportive care.",
        key_recommendations=[
            "Endoscopic mucosal resection (EMR) or endoscopic submucosal dissection (ESD) for carefully selected T1a well-differentiated non-ulcerated lesions < 2 cm.",
            "Subtotal or total gastrectomy with D2 lymph node dissection is the surgical standard with curative intent for locoregional disease."
        ]
    ),
    "PAN_ASIAN_ESMO_2024": EvidenceSource(
        source_id="PAN_ASIAN_ESMO_2024",
        title="Pan-Asian Adapted ESMO Clinical Practice Guidelines for the Management of Patients with Gastric Cancer",
        organization="ESMO / JSMO / KSMO / CSCO / TOS",
        publication_date="2024-03-15",
        version_or_edition="Pan-Asian Consensus",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10937212/",
        evidence_level=EvidenceLevel.GUIDELINE,
        jurisdiction=Jurisdiction.GLOBAL,
        doi_or_pmid="10.1016/j.annonc.2024.01.006",
        summary="Consensus adaptation of ESMO guidelines reflecting Asian surgical standards (routine D2 lymphadenectomy) and regional chemotherapy regimens (S-1 based).",
        key_recommendations=[
            "Routine D2 lymphadenectomy endorsed as standard of care in high-volume Asian medical centers.",
            "Adjuvant S-1 (or S-1 plus oxaliplatin / capecitabine plus oxaliplatin) recommended after D2 gastrectomy for pathological stage II/III."
        ]
    ),
    "KOREAN_2024": EvidenceSource(
        source_id="KOREAN_2024",
        title="Korean Practice Guideline for Gastric Cancer 2024: An Evidence-based, Multidisciplinary Approach",
        organization="Korean Gastric Cancer Association (KGCA)",
        publication_date="2025-01-20",
        version_or_edition="2024 Update (JGC 2025)",
        url="https://www.jgc-online.org/DOIx.php?id=10.5230/jgc.2025.25.e11",
        evidence_level=EvidenceLevel.GUIDELINE,
        jurisdiction=Jurisdiction.SOUTH_KOREA,
        doi_or_pmid="10.5230/jgc.2025.25.e11",
        summary="Multidisciplinary guidelines for early detection, laparoscopic/robotic gastrectomy, and adjuvant regimens in East Asian cohorts.",
        key_recommendations=[
            "Biennial endoscopic screening for adults aged 40-75.",
            "Minimally invasive gastrectomy for early gastric cancer."
        ]
    ),
}

# ---------------------------------------------------------------------------
# Pathology & Diagnostic Standards
# ---------------------------------------------------------------------------
PATHOLOGY_STANDARDS: dict[str, EvidenceSource] = {
    "WHO_6TH_ED_2026": EvidenceSource(
        source_id="WHO_6TH_ED_2026",
        title="WHO Classification of Tumours, 6th Edition: Digestive System Tumours",
        organization="World Health Organization (WHO) / International Agency for Research on Cancer (IARC)",
        publication_date="2026-04-10",
        version_or_edition="6th Edition (Blue Book)",
        url="https://publications.iarc.who.int/662",
        evidence_level=EvidenceLevel.PATHOLOGY_STANDARD,
        jurisdiction=Jurisdiction.GLOBAL,
        summary="Gold standard international histological classification of gastric epithelial neoplasms, neuroendocrine tumors, mesenchymal tumors, and lymphomas.",
        key_recommendations=[
            "Gastric adenocarcinoma classified into Tubular, Papillary, Mucinous, Poorly Cohesive (including signet-ring cell type), and Mixed types.",
            "Explicit integration of molecular subtyping: EBV+, MSI-H, GS (Genomically Stable), and CIN (Chromosomal Instability).",
            "Clear distinction of adenocarcinoma from GIST, lymphoma (MALT/DLBCL), and neuroendocrine neoplasms (NET G1-G3, NEC)."
        ]
    ),
    "CAP_STOMACH_2026": EvidenceSource(
        source_id="CAP_STOMACH_2026",
        title="Protocol for the Examination of Specimens From Patients With Carcinoma of the Stomach",
        organization="College of American Pathologists (CAP)",
        publication_date="2026-07-01",
        version_or_edition="Version 4.4.0.0 REL CAPCP",
        url="https://www.cap.org/protocols-and-guidelines/cancer-protocols/current-cancer-protocols/",
        evidence_level=EvidenceLevel.PATHOLOGY_STANDARD,
        jurisdiction=Jurisdiction.US,
        summary="Authoritative synoptic pathology reporting protocol for gastric biopsy, endoscopic resection, and partial/total gastrectomy specimens.",
        key_recommendations=[
            "Mandatory recording of histological type, Lauren classification, histological grade, depth of invasion (T), lymph node status (N), margins, LVI, and PNI.",
            "CAP element-response pairs required for all synoptic resection reports.",
            "Post-neoadjuvant specimens require documentation of treatment effect and Tumor Regression Grade (Ryan / AJCC system)."
        ]
    ),
    "ICCR_STOMACH_2024": EvidenceSource(
        source_id="ICCR_STOMACH_2024",
        title="International Collaboration on Cancer Reporting: Stomach Carcinoma Dataset",
        organization="ICCR",
        publication_date="2024-05-01",
        version_or_edition="2nd Edition",
        url="https://www.iccr-cancer.org/datasets/digestive-tract/stomach/",
        evidence_level=EvidenceLevel.PATHOLOGY_STANDARD,
        jurisdiction=Jurisdiction.GLOBAL,
        summary="Internationally harmonized dataset specification for reporting gastric cancer resections.",
        key_recommendations=[
            "Core data items: procedure, tumor site, histological type, grade, macroscopic type, invasion depth, margins, lymph nodes (>=16 recommended)."
        ]
    ),
}

# ---------------------------------------------------------------------------
# Biomarker Companion Diagnostic Specifications
# ---------------------------------------------------------------------------
BIOMARKER_STANDARDS: dict[str, dict[str, Any]] = {
    "HER2": {
        "biomarker": "HER2 / ERBB2",
        "method": "Immunohistochemistry (IHC) and in-situ hybridization (FISH/CISH)",
        "interpretation_criteria": "Gastric-Specific Hofmann Criteria (Biopsy: cluster of >= 5 cells; Resection: >= 10% cells)",
        "positive_definition": "IHC 3+ (strong basolateral/membranous staining) OR IHC 2+ with positive ISH (HER2:CEP17 ratio >= 2.0)",
        "clinical_indication": "Predictive of benefit from Trastuzumab (1st line with chemo) and Trastuzumab Deruxtecan (T-DXd, 2nd/3rd line).",
        "jurisdictional_approval": {
            Jurisdiction.US: "FDA Approved (HercepTest, PATHWAY anti-HER2/neu 4B5; T-DXd in >= 2nd line)",
            Jurisdiction.EU: "EMA Approved (Trastuzumab + chemo in IHC3+ or IHC2+/FISH+; T-DXd post-trastuzumab)",
            Jurisdiction.INDIA: "CDSCO Approved (Trastuzumab biosimilars widely available; T-DXd approved for HER2+ advanced GC)",
            Jurisdiction.JAPAN: "PMDA Approved (Trastuzumab 1st line; T-DXd in advanced/recurrent GC)"
        }
    },
    "PD_L1": {
        "biomarker": "PD-L1 (Programmed Death-Ligand 1)",
        "method": "Immunohistochemistry using verified companion assays (e.g., 28-8 pharmDx, 22C3 pharmDx)",
        "score_metric": "Combined Positive Score (CPS) = (PD-L1+ tumor cells + lymphocytes + macrophages / Total viable tumor cells) * 100",
        "interpretation_thresholds": {
            "CPS >= 1": "Permissive for pembrolizumab combination in specific jurisdictional labels.",
            "CPS >= 5": "Standard threshold for Nivolumab + chemo in ESMO guidelines (CheckMate 649).",
            "CPS >= 10": "High-benefit threshold in ASCO / FDA guidelines for Pembrolizumab + chemo (KEYNOTE-859)."
        },
        "jurisdictional_approval": {
            Jurisdiction.US: "FDA Approved: Nivolumab + chemo (regardless of PD-L1, but greatest benefit CPS >= 5); Pembrolizumab + chemo (CPS >= 1)",
            Jurisdiction.EU: "EMA Approved: Nivolumab + chemo restricted to PD-L1 CPS >= 5; Pembrolizumab + chemo restricted to CPS >= 1 or >= 10 depending on indication",
            Jurisdiction.INDIA: "CDSCO Approved: Nivolumab and Pembrolizumab registered; CPS testing standard in specialized tertiary cancer centers (e.g., TMH)",
            Jurisdiction.JAPAN: "PMDA Approved: Nivolumab + chemo approved first-line without formal CPS cut-off restriction"
        }
    },
    "MMR_MSI": {
        "biomarker": "Mismatch Repair (MMR) & Microsatellite Instability (MSI)",
        "method": "IHC for MLH1, PMS2, MSH2, MSH6; or PCR/NGS for MSI status",
        "interpretation_criteria": "dMMR: Complete loss of nuclear expression in >= 1 MMR protein with intact internal control. MSI-H: Instability in >= 2 of 5 mononucleotide repeats.",
        "clinical_indication": "Predictive of marked sensitivity to immune checkpoint inhibitors (Pembrolizumab, Nivolumab +/- Ipilimumab) and potential resistance or lack of benefit from adjuvant fluoropyrimidine monotherapy.",
        "jurisdictional_approval": {
            Jurisdiction.US: "FDA Approved: Tumor-agnostic pembrolizumab or dostarlimab for dMMR/MSI-H solid tumors.",
            Jurisdiction.EU: "EMA Approved: Pembrolizumab monotherapy for dMMR/MSI-H gastric cancer after prior therapy.",
            Jurisdiction.INDIA: "CDSCO Approved: Pembrolizumab approved for MSI-H/dMMR solid tumors progressing on standard therapy."
        }
    },
    "CLDN18_2": {
        "biomarker": "Claudin 18.2 (CLDN18.2)",
        "method": "Immunohistochemistry using VENTANA CLDN18 (43-14A) RxDx Assay",
        "interpretation_criteria": "SPOTLIGHT & GLOW Trial Criteria: >= 75% of viable tumor cells displaying moderate-to-strong (>= 2+) membranous staining.",
        "clinical_indication": "Predictive of benefit from Zolbetuximab (anti-CLDN18.2 monoclonal antibody) combined with mFOLFOX6 or CAPOX.",
        "jurisdictional_approval": {
            Jurisdiction.US: "FDA Approved: Zolbetuximab (Vyloy) + chemo for CLDN18.2+, HER2- advanced/metastatic gastric adenocarcinoma (approved 2024).",
            Jurisdiction.EU: "EMA Approved: Zolbetuximab approved late 2024 / 2025 for first-line CLDN18.2+ HER2- metastatic GC.",
            Jurisdiction.JAPAN: "PMDA Approved: First global approval in March 2024 for CLDN18.2+ advanced gastric cancer.",
            Jurisdiction.INDIA: "CDSCO Status: Import under named-patient access / compassionate use; formal commercial launch undergoing regulatory review (2026). Clearly disclosed as restricted commercial availability."
        }
    }
}

# ---------------------------------------------------------------------------
# Research Reporting Standards
# ---------------------------------------------------------------------------
REPORTING_STANDARDS: dict[str, EvidenceSource] = {
    "CLAIM_2024": EvidenceSource(
        source_id="CLAIM_2024",
        title="Checklist for Artificial Intelligence in Medical Imaging (CLAIM) 2024 Update",
        organization="Radiological Society of North America (RSNA) / Radiology: Artificial Intelligence",
        publication_date="2024-05-29",
        version_or_edition="CLAIM 2024",
        url="https://pubmed.ncbi.nlm.nih.gov/38809149/",
        evidence_level=EvidenceLevel.PATHOLOGY_STANDARD,
        jurisdiction=Jurisdiction.GLOBAL,
        doi_or_pmid="10.1148/ryai.240166",
        summary="Authoritative 44-item checklist establishing rigorous standards for reporting AI model architectures, data partitioning, leakage prevention, and clinical validity."
    ),
    "TRIPOD_AI_2024": EvidenceSource(
        source_id="TRIPOD_AI_2024",
        title="TRIPOD+AI Statement: Reporting of Clinical Prediction Models Developed or Evaluated Using Machine Learning",
        organization="Equator Network / BMJ",
        publication_date="2024-04-16",
        version_or_edition="TRIPOD+AI 2024",
        url="https://www.bmj.com/content/385/bmj-2023-078378",
        evidence_level=EvidenceLevel.PATHOLOGY_STANDARD,
        jurisdiction=Jurisdiction.GLOBAL,
        doi_or_pmid="10.1136/bmj-2023-078378",
        summary="Standardized reporting guidelines for multivariable diagnostic and prognostic machine learning prediction models in medicine."
    ),
    "PRISMA_2020": EvidenceSource(
        source_id="PRISMA_2020",
        title="The PRISMA 2020 Statement: An Updated Guideline for Reporting Systematic Reviews",
        organization="PRISMA Statement",
        publication_date="2021-03-29",
        version_or_edition="PRISMA 2020",
        url="https://www.prisma-statement.org/prisma-2020-statement",
        evidence_level=EvidenceLevel.PATHOLOGY_STANDARD,
        jurisdiction=Jurisdiction.GLOBAL,
        doi_or_pmid="10.1136/bmj.n71",
        summary="27-item checklist and 4-phase flow diagram methodology for reporting systematic literature reviews and meta-analyses."
    ),
}


def get_evidence_source(source_id: str) -> Optional[EvidenceSource]:
    """Retrieves an evidence source by identifier."""
    if source_id in GUIDELINE_REGISTRY:
        return GUIDELINE_REGISTRY[source_id]
    if source_id in PATHOLOGY_STANDARDS:
        return PATHOLOGY_STANDARDS[source_id]
    if source_id in REPORTING_STANDARDS:
        return REPORTING_STANDARDS[source_id]
    return None


def get_guidelines_for_jurisdiction(jurisdiction: Jurisdiction) -> list[EvidenceSource]:
    """Filters clinical guidelines applicable to a specific jurisdiction."""
    results = []
    for source in GUIDELINE_REGISTRY.values():
        if source.jurisdiction in (jurisdiction, Jurisdiction.GLOBAL):
            results.append(source)
    return results


def get_biomarker_context(biomarker_name: str, jurisdiction: Jurisdiction = Jurisdiction.GLOBAL) -> dict[str, Any]:
    """Retrieves interpretation criteria and jurisdiction-specific approvals for a biomarker."""
    key = biomarker_name.upper().replace("-", "_").replace(".", "_")
    if key not in BIOMARKER_STANDARDS:
        # Search partial matches
        for b_key, b_data in BIOMARKER_STANDARDS.items():
            if b_key in key or key in b_key:
                key = b_key
                break
    if key in BIOMARKER_STANDARDS:
        data = dict(BIOMARKER_STANDARDS[key])
        approvals = data.get("jurisdictional_approval", {})
        data["local_approval"] = approvals.get(jurisdiction, approvals.get(Jurisdiction.US, "Information not documented for this jurisdiction."))
        return data
    return {
        "biomarker": biomarker_name,
        "status": "Unrecognized or exploratory biomarker",
        "guideline_context": "Not documented in standard clinical practice guidelines for primary routine gastric cancer management."
    }

