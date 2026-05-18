"""
╔══════════════════════════════════════════════════════════════╗
║         PIPELINE INTEGRADO — OLIST E-COMMERCE DATASET       ║
║         Tech Challenge Fase 1 — POSTECH DTAT                ║
╠══════════════════════════════════════════════════════════════╣
║  Como executar:                                              ║
║    python pipeline_olist.py                                  ║
║                                                              ║
║  O pipeline tem 5 etapas sequenciais:                        ║
║    [1] Ingestão    — leitura dos CSVs                        ║
║    [2] Qualidade   — tratamento de nulos                     ║
║    [3] Enriquecimento — joins e colunas derivadas            ║
║    [4] Análises    — todos os indicadores do projeto         ║
║    [5] Exportação  — resultados em CSV prontos para uso      ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO CENTRAL
# Altere apenas aqui se os arquivos estiverem em outro diretório
# ──────────────────────────────────────────────────────────────
DATA_DIR   = "."          # diretório dos CSVs de entrada
OUTPUT_DIR = "./outputs"  # diretório dos resultados gerados

# ══════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════

def log(etapa, mensagem):
    """Imprime log padronizado com timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{etapa}] {mensagem}")

def salvar(df, nome_arquivo, etapa="EXPORT"):
    """Salva DataFrame em CSV no diretório de saída."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    caminho = os.path.join(OUTPUT_DIR, nome_arquivo)
    df.to_csv(caminho, index=False, encoding='utf-8-sig')
    log(etapa, f"Salvo: {nome_arquivo} ({len(df):,} linhas)")

def separador(titulo):
    print(f"\n{'═'*60}")
    print(f"  {titulo}")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════
# ETAPA 1 — INGESTÃO
# Carrega todos os CSVs brutos sem qualquer transformação.
# Objetivo: ter os dados exatamente como vieram da fonte.
# ══════════════════════════════════════════════════════════════

def etapa_ingestion(data_dir):
    separador("ETAPA 1 — INGESTÃO DOS DADOS")

    arquivos = {
        "orders"     : "olist_orders_dataset.csv",
        "payments"   : "olist_order_payments_dataset.csv",
        "reviews"    : "olist_order_reviews_dataset.csv",
        "items"      : "olist_order_items_dataset.csv",
        "customers"  : "olist_customers_dataset.csv",
        "products"   : "olist_products_dataset.csv",
        "sellers"    : "olist_sellers_dataset.csv",
        "translation": "product_category_name_translation.csv",
    }

    dfs = {}
    for nome, arquivo in arquivos.items():
        caminho = os.path.join(data_dir, arquivo)
        dfs[nome] = pd.read_csv(caminho)
        log("INGESTION", f"{nome:12s} → {len(dfs[nome]):>7,} linhas | {dfs[nome].shape[1]} colunas")

    return dfs


# ══════════════════════════════════════════════════════════════
# ETAPA 2 — QUALIDADE DE DADOS (Tratamento de Nulos)
# Cada decisão está documentada inline.
# Regra geral: nulo semântico é mantido; nulo por erro é corrigido.
# ══════════════════════════════════════════════════════════════

def etapa_quality(dfs):
    separador("ETAPA 2 — QUALIDADE DE DADOS")

    # ── 2.1 ORDERS ────────────────────────────────────────────
    log("QUALITY", "orders → convertendo datas...")
    orders = dfs["orders"].copy()

    colunas_data = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    for col in colunas_data:
        orders[col] = pd.to_datetime(orders[col], errors="coerce").astype("datetime64[us]")

    # order_approved_at: 14 pedidos 'delivered' sem data de aprovação
    # → anomalia de registro; imputar com mediana do tempo compra→aprovação
    tempo_h = (
        (orders["order_approved_at"] - orders["order_purchase_timestamp"])
        .dt.total_seconds() / 3600
    )
    mediana_aprov_h = tempo_h.median()

    mask_anomalia = (
        orders["order_approved_at"].isnull() &
        (orders["order_status"] == "delivered")
    )
    imputado = (
        orders.loc[mask_anomalia, "order_purchase_timestamp"]
        + pd.to_timedelta(mediana_aprov_h, unit="h")
    ).astype("datetime64[us]")
    orders.loc[mask_anomalia, "order_approved_at"] = imputado

    log("QUALITY", f"orders → 14 aprovações de pedidos 'delivered' imputadas "
                   f"(mediana={mediana_aprov_h:.1f}h)")
    log("QUALITY", "orders → demais nulos MANTIDOS (nulo semântico: pedidos não entregues)")

    dfs["orders"] = orders

    # ── 2.2 REVIEWS ───────────────────────────────────────────
    reviews = dfs["reviews"].copy()
    antes = reviews.isnull().sum().sum()

    # Campos de texto opcionais → preencher com placeholder
    # para evitar erros em operações de string
    reviews["review_comment_title"]   = reviews["review_comment_title"].fillna("Sem título")
    reviews["review_comment_message"] = reviews["review_comment_message"].fillna("Sem comentário")

    depois = reviews.isnull().sum().sum()
    log("QUALITY", f"reviews → {antes - depois:,} nulos substituídos por placeholders de texto")
    dfs["reviews"] = reviews

    # ── 2.3 PRODUCTS ──────────────────────────────────────────
    products = dfs["products"].copy()
    antes_p = products.isnull().sum().sum()

    # Bloco A: 610 produtos sem cadastro completo (mesmas linhas)
    products["product_category_name"] = products["product_category_name"].fillna("sem_categoria")

    for col in ["product_name_lenght", "product_description_lenght", "product_photos_qty"]:
        mediana = products[col].median()
        products[col] = products[col].fillna(mediana)

    # Bloco B: 2 produtos sem dimensões físicas
    for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        mediana = products[col].median()
        products[col] = products[col].fillna(mediana)

    depois_p = products.isnull().sum().sum()
    log("QUALITY", f"products → {antes_p - depois_p} nulos imputados com mediana ou placeholder")
    dfs["products"] = products

    # ── 2.4 DATASETS SEM NULOS (sem ação) ─────────────────────
    for nome in ["payments", "items", "customers", "sellers", "translation"]:
        log("QUALITY", f"{nome:12s} → ✅ sem nulos (nenhuma ação necessária)")

    return dfs


# ══════════════════════════════════════════════════════════════
# ETAPA 3 — ENRIQUECIMENTO
# Cria colunas derivadas e tabelas analíticas unificadas.
# Tudo que é join ou cálculo de suporte vai aqui.
# ══════════════════════════════════════════════════════════════

def etapa_enrichment(dfs):
    separador("ETAPA 3 — ENRIQUECIMENTO DOS DADOS")

    orders   = dfs["orders"]
    items    = dfs["items"].copy()
    products = dfs["products"]
    translation = dfs["translation"]
    payments = dfs["payments"]
    customers = dfs["customers"]

    # ── 3.1 Tradução de categorias ────────────────────────────
    cat_map = translation.set_index("product_category_name")["product_category_name_english"].to_dict()
    products = products.copy()
    products["category_en"] = products["product_category_name"].map(cat_map).fillna("uncategorized")
    log("ENRICHMENT", "Categorias traduzidas para inglês (translation join)")

    # ── 3.2 Razão frete/preço ─────────────────────────────────
    items["freight_ratio"] = items["freight_value"] / items["price"].replace(0, np.nan)
    items["frete_alto"]    = items["freight_ratio"] >= 0.75
    log("ENRICHMENT", "Coluna 'freight_ratio' e flag 'frete_alto' criadas em items")

    # ── 3.3 Tempo de entrega e atraso ─────────────────────────
    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered["lead_time_dias"] = (
        delivered["order_delivered_customer_date"] -
        delivered["order_purchase_timestamp"]
    ).dt.days
    delivered["atraso_dias"] = (
        delivered["order_delivered_customer_date"] -
        delivered["order_estimated_delivery_date"]
    ).dt.days
    delivered["entregue_atrasado"] = delivered["atraso_dias"] > 0
    log("ENRICHMENT", "Colunas 'lead_time_dias', 'atraso_dias', 'entregue_atrasado' criadas")

    # ── 3.4 Período (ano_mes) nos pedidos ─────────────────────
    orders = orders.copy()
    orders["ano_mes"] = orders["order_purchase_timestamp"].dt.to_period("M")
    log("ENRICHMENT", "Coluna 'ano_mes' criada para séries temporais")

    # ── 3.5 Tabela mestre: pedido + entrega + review ──────────
    reviews = dfs["reviews"][["order_id", "review_score"]].copy()
    mestre = delivered.merge(reviews, on="order_id", how="left")
    mestre = mestre.merge(customers[["customer_id", "customer_state", "customer_unique_id"]],
                          on="customer_id", how="left")
    log("ENRICHMENT", "Tabela mestre criada: orders + reviews + customers (apenas 'delivered')")

    dfs["orders"]    = orders
    dfs["items"]     = items
    dfs["products"]  = products
    dfs["delivered"] = delivered
    dfs["mestre"]    = mestre

    return dfs


# ══════════════════════════════════════════════════════════════
# ETAPA 4 — ANÁLISES
# Cada bloco corresponde a uma pergunta de negócio.
# Todos os scripts individuais do projeto estão consolidados aqui.
# ══════════════════════════════════════════════════════════════

def etapa_analysis(dfs):
    separador("ETAPA 4 — ANÁLISES DE NEGÓCIO")

    orders   = dfs["orders"]
    payments = dfs["payments"]
    reviews  = dfs["reviews"]
    items    = dfs["items"]
    products = dfs["products"]
    delivered = dfs["delivered"]
    mestre   = dfs["mestre"]
    customers = dfs["customers"]

    resultados = {}

    # ── A. GMV e Faturamento Total ────────────────────────────
    log("ANALYSIS", "A. Faturamento (GMV)...")
    gmv = payments["payment_value"].sum()
    ticket_medio = payments.groupby("order_id")["payment_value"].sum().mean()
    ticket_mediana = payments.groupby("order_id")["payment_value"].sum().median()

    resultados["A_gmv"] = pd.DataFrame([{
        "GMV_total_R$"     : round(gmv, 2),
        "ticket_medio_R$"  : round(ticket_medio, 2),
        "ticket_mediana_R$": round(ticket_mediana, 2),
        "total_pedidos"    : orders["order_id"].nunique(),
        "total_clientes"   : customers["customer_unique_id"].nunique(),
        "total_vendedores" : dfs["sellers"]["seller_id"].nunique()
    }])

    # ── B. Evolução Mensal ────────────────────────────────────
    log("ANALYSIS", "B. Evolução mensal de pedidos e receita...")
    pag_order = payments.groupby("order_id")["payment_value"].sum().reset_index()
    orders_pag = orders.merge(pag_order, on="order_id", how="inner")

    evolucao = orders_pag.groupby("ano_mes").agg(
        pedidos   =("order_id", "count"),
        receita   =("payment_value", "sum")
    ).reset_index()
    evolucao["ticket_medio"] = (evolucao["receita"] / evolucao["pedidos"]).round(2)
    evolucao["receita"]      = evolucao["receita"].round(2)
    evolucao = evolucao[evolucao["ano_mes"].astype(str).between("2017-01", "2018-08")]
    resultados["B_evolucao_mensal"] = evolucao

    # ── C. Meios de Pagamento ─────────────────────────────────
    log("ANALYSIS", "C. Meios de pagamento...")
    pgmt = payments.groupby("payment_type").agg(
        transacoes  =("payment_value", "count"),
        valor_total =("payment_value", "sum")
    ).reset_index()
    pgmt["pct_valor"] = (pgmt["valor_total"] / pgmt["valor_total"].sum() * 100).round(2)
    pgmt = pgmt.sort_values("valor_total", ascending=False)
    resultados["C_meios_pagamento"] = pgmt

    # ── D. Cartão: à vista vs parcelado ──────────────────────
    log("ANALYSIS", "D. Cartão de crédito: à vista vs parcelado...")
    credito = payments[payments["payment_type"] == "credit_card"].copy()
    credito["tipo_parcelamento"] = credito["payment_installments"].apply(
        lambda x: "Parcelado (>1x)" if x > 1 else "À Vista (1x)"
    )
    tabela_credito = credito["tipo_parcelamento"].value_counts().reset_index()
    tabela_credito.columns = ["tipo", "transacoes"]
    tabela_credito["pct"] = (tabela_credito["transacoes"] / tabela_credito["transacoes"].sum() * 100).round(2)
    resultados["D_credito_vista_parcelado"] = tabela_credito

    # ── E. Recebíveis longos (≥4 parcelas) ───────────────────
    log("ANALYSIS", "E. Recebíveis longos (crédito ≥ 4 parcelas)...")
    filtro_longo = (
        (payments["payment_type"] == "credit_card") &
        (payments["payment_installments"] >= 4)
    )
    seg = payments[filtro_longo]
    resultados["E_recebiveis_longos"] = pd.DataFrame([{
        "valor_R$"          : round(seg["payment_value"].sum(), 2),
        "pct_do_GMV"        : round(seg["payment_value"].sum() / payments["payment_value"].sum() * 100, 2),
        "transacoes"        : len(seg),
        "pct_transacoes"    : round(len(seg) / len(payments) * 100, 2)
    }])

    # ── F. Distribuição de Reviews ───────────────────────────
    log("ANALYSIS", "F. Distribuição de notas...")
    dist_notas = reviews["review_score"].value_counts().sort_index(ascending=False).reset_index()
    dist_notas.columns = ["nota", "qtd"]
    dist_notas["pct"] = (dist_notas["qtd"] / dist_notas["qtd"].sum() * 100).round(2)
    resultados["F_distribuicao_notas"] = dist_notas

    # ── G. Nota média: atrasados vs no prazo ─────────────────
    log("ANALYSIS", "G. Correlação atraso x nota...")
    corr = mestre.groupby("entregue_atrasado")["review_score"].mean().round(2).reset_index()
    corr.columns = ["entregue_atrasado", "nota_media"]
    corr["status"] = corr["entregue_atrasado"].map({True: "Atrasado", False: "No Prazo"})
    resultados["G_atraso_x_nota"] = corr[["status", "nota_media"]]

    # ── H. Atraso na entrega ─────────────────────────────────
    log("ANALYSIS", "H. Status de entrega...")
    total_del = len(delivered)
    atrasados = delivered["entregue_atrasado"].sum()
    atraso_medio = delivered.loc[delivered["entregue_atrasado"], "atraso_dias"].mean()
    resultados["H_status_entrega"] = pd.DataFrame([{
        "total_pedidos_entregues" : total_del,
        "entregues_no_prazo"      : total_del - atrasados,
        "entregues_atrasados"     : atrasados,
        "pct_atrasados"           : round(atrasados / total_del * 100, 2),
        "atraso_medio_dias"       : round(atraso_medio, 1)
    }])

    # ── I. Frete alto ────────────────────────────────────────
    log("ANALYSIS", "I. Custo de frete...")
    frete = items["frete_alto"].value_counts().reset_index()
    frete.columns = ["frete_alto", "qtd_itens"]
    frete["pct"] = (frete["qtd_itens"] / frete["qtd_itens"].sum() * 100).round(2)
    resultados["I_frete_alto"] = frete

    # ── J. Top 10 categorias por receita ─────────────────────
    log("ANALYSIS", "J. Receita por categoria...")
    items_cat = items.merge(products[["product_id", "category_en"]], on="product_id", how="left")
    cat_receita = items_cat.groupby("category_en")["price"].sum().sort_values(ascending=False).head(10).reset_index()
    cat_receita.columns = ["categoria", "receita_R$"]
    cat_receita["pct"] = (cat_receita["receita_R$"] / items_cat["price"].sum() * 100).round(2)
    resultados["J_top10_categorias"] = cat_receita

    # ── K. Recompra ──────────────────────────────────────────
    log("ANALYSIS", "K. Taxa de recompra...")
    compras_por_cliente = (
        orders.merge(customers, on="customer_id")
        .groupby("customer_unique_id")["order_id"].nunique()
    )
    recompra = (compras_por_cliente > 1).sum()
    total_cli = compras_por_cliente.shape[0]
    resultados["K_recompra"] = pd.DataFrame([{
        "total_clientes_unicos" : total_cli,
        "clientes_com_recompra" : recompra,
        "taxa_recompra_pct"     : round(recompra / total_cli * 100, 2)
    }])

    # ── L. Concentração geográfica ───────────────────────────
    log("ANALYSIS", "L. Distribuição por estado...")
    geo = (
        orders.merge(customers, on="customer_id")
        ["customer_state"].value_counts()
        .reset_index()
    )
    geo.columns = ["estado", "pedidos"]
    geo["pct"] = (geo["pedidos"] / geo["pedidos"].sum() * 100).round(2)
    resultados["L_distribuicao_estados"] = geo

    # ── M. Motivo das notas 1 ────────────────────────────────
    log("ANALYSIS", "M. Motivo das avaliações nota 1...")
    notas_1 = reviews[reviews["review_score"] == 1].copy()
    notas_1 = notas_1[notas_1["review_comment_message"] != "Sem comentário"].copy()
    notas_1["motivo"] = "Outros / Sem detalhe suficiente"
    notas_1.loc[
        notas_1["review_comment_message"].str.contains(
            "não recebi|não chegou|entrega|atras|demorou|prazo", na=False, case=False
        ), "motivo"
    ] = "Problemas Logísticos"
    notas_1.loc[
        notas_1["review_comment_message"].str.contains(
            "diferente|errado|não era|anúncio|descrição|falso", na=False, case=False
        ), "motivo"
    ] = "Produto Incorreto / Divergente"
    notas_1.loc[
        notas_1["review_comment_message"].str.contains(
            "quebrado|defeito|danificado|parou|não funciona|estragado", na=False, case=False
        ), "motivo"
    ] = "Produto com Defeito"
    notas_1.loc[
        notas_1["review_comment_message"].str.contains(
            "não respondeu|vendedor|atendimento|suporte|ignorou", na=False, case=False
        ), "motivo"
    ] = "Atendimento do Vendedor"

    motivos = notas_1["motivo"].value_counts().reset_index()
    motivos.columns = ["motivo", "qtd"]
    motivos["pct"] = (motivos["qtd"] / motivos["qtd"].sum() * 100).round(2)
    resultados["M_motivos_nota1"] = motivos

    return resultados


# ══════════════════════════════════════════════════════════════
# ETAPA 5 — EXPORTAÇÃO
# Salva cada resultado em CSV individual no diretório /outputs.
# Convenção de nome: {letra}_{descricao}.csv
# ══════════════════════════════════════════════════════════════

def etapa_export(resultados):
    separador("ETAPA 5 — EXPORTAÇÃO DOS RESULTADOS")

    for chave, df in resultados.items():
        salvar(df, f"{chave}.csv")

    log("EXPORT", f"✅ {len(resultados)} arquivos salvos em '{OUTPUT_DIR}/'")


# ══════════════════════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════

def main():
    inicio = datetime.now()
    print(f"\n{'═'*60}")
    print(f"  PIPELINE INICIADO — {inicio.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{'═'*60}")

    dfs        = etapa_ingestion(DATA_DIR)
    dfs        = etapa_quality(dfs)
    dfs        = etapa_enrichment(dfs)
    resultados = etapa_analysis(dfs)
    etapa_export(resultados)

    fim = datetime.now()
    duracao = (fim - inicio).seconds
    separador("PIPELINE CONCLUÍDO")
    print(f"  Duração total : {duracao}s")
    print(f"  Resultados em : {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  Arquivos gerados: {len(resultados)}")
    print()


if __name__ == "__main__":
    main()
