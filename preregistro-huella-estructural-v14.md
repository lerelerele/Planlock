# Preregistro v14 — Estudio piloto de estabilidad de la huella estructural

**Estado:** BORRADOR. **Listo para ejecutar E0.** La firma y el hash final
esperan al resultado de E0 — validez real de los PEs bajo grupos distribuidos
reales. La decisión sobre `routed_item` (identidad 19) quedó cerrada en el
dry-run de calibración sintético: ver §1.6.3, §1.6.4.B y §8.3.5.
**Repositorio objetivo:** pytorch/torchtitan
**HEAD de referencia:** `9a711521ac2973fe230a3f38efc6aedfc7d1f9c6`
**Ventana:** `[2026-05-17T17:00:00Z, 2026-08-15T17:00:00Z)`
**Timebox:** 5 días de trabajo.

Cambios sobre v13: `op_comunicacion` en la firma del clasificador y **colisión
`NONE`/`SendRecv` resuelta** (§1.4), **unidad de conteo de `SendRecv`** (§1.4),
referencia expresa a la política de fases (§1.4), y **restauradas** la
serialización canónica de `forma_normalizada` (§1.6.3) y la definición de
equivalencia de Q1 (§1.8).

---

## 0. Alcance epistemológico

**Puede:** falsar la hipótesis de que la huella es estable bajo refactors
reales. Un solo falso positivo intrínseco basta.

**No puede:** confirmar estabilidad (0/20 acota la FPR en **13,9%**, límite
superior unilateral exacto al 95%); establecer el SLO; demostrar
extraibilidad automática (E4).

### 0.1 Tres cantidades, nunca reportadas por separado

| Cantidad | Definición |
|---|---|
| **FPR** | `FP_PR` / negativos — lo que mide el piloto |
| **Tasa operacional** | `FP_GATE` / PRs cualificantes — define el SLO; en el piloto solo un **intervalo** |
| **Cobertura** | §8.2 |

### 0.2 Fase sombra

```
n = ⌈ log(0.05) / log(136/137) ⌉ = 409
```

409 PRs cualificantes consecutivos sin falsa alarma acotan la tasa operacional
en 0,73%: **38,4 semanas**. Acotar la FPR igual exigiría 409 *negativos*, no
409 PRs. **Termina por tamaño muestral, no por calendario:** N ≥ 409 bajo una
versión congelada del checker, mínimo un trimestre; cualquier cambio reinicia
el contador. **Es la única medición válida del SLO.**

---

## 1. Definición de la huella (INMUTABLE tras la firma)

### 1.0 Punto de evaluación (PE)

```
PE := (
    modulo_registro, funcion_config, overrides, hash_manifiesto,
    grados, malla_densa, malla_dispersa, particion_pp,
    identidades_arquitectonicas,        # §1.6.2
    backend, frontera
)
```

En este HEAD la configuración procede de funciones Python de `config_registry`,
no de un `.toml`. `particion_pp` debe ser explícita porque §1.8 la deja fuera
del cociente.

**Validez del PE — activación real del eje:**

> Un eje está **cubierto** solo si tiene grado mayor que uno **y** el plan
> contiene al menos una plantilla con transición distinta de `NONE` cuyo
> `grupo_comunicacion` sea ese eje. Si algún eje en alcance no está cubierto,
> **el PE es inválido**.

Grado > 1 es necesario y no suficiente. Comprobación sobre **grupos reales,
incluido `efsdp`**. Ejes excluidos: `⊥` (OFF), **nunca `1`**.

**Grados y dominios.** `ep` reutiliza los ranks densos: no multiplica el world
size. `efsdp = dp_s × cp × tp / ep`.

| Eje | `PE_dense` | `PE_moe` |
|---|---|---|
| `dp_r` | 2 | ⊥ |
| `dp_s` | 2 | 2 |
| `cp` | 2 | ⊥ |
| `tp` | 2 | 2 |
| `ep` | ⊥ | 2 |
| `efsdp` | n/a | 2 (derivado) |
| `pp` | 2 | 2 |
| **world size** | **32** | **8** |

```
PE_dense: 2 × 2 × 2 × 2 × 2 = 32
PE_moe:   1 × 2 × 1 × 2 × 2 = 8       (dp_r × dp_s × cp × tp × pp)
          efsdp = 2 × 1 × 2 / 2 = 2
```

| | `malla_densa` | `malla_dispersa` |
|---|---|---|
| `PE_dense` | `{dp_r, dp_s, cp, tp}` | n/a |
| `PE_moe` | `{dp_s, tp}` | `{efsdp, ep}` |

**`pp` está cubierto pero no pertenece a ningún dominio de placement.**
`PE_dense` con `dp_r=2 ∧ dp_s=2` es HSDP. Los grados son **propuesta no
verificada**; E0 (§8.3) los valida.

### 1.1 Plantilla y huella (definición formal)

```
plantilla := (
    rol_productor,          # §1.3
    placement_productor,    # §1.5
    transicion,             # §1.4
    placement_consumidor,   # §1.5
    rol_consumidor,         # §1.3
    firma_tensor,           # §1.6
    grupo_comunicacion      # §1.5
)

huella := mapa  plantilla → multiplicidad      # §1.7
```

Estos son **los siete campos** a los que se refiere `spec_suficiente` (§10.3),
lo que devuelve `Huella(C, lado)` (§5.2), y aquello sobre lo que opera `⊖`.

**Diferencia simétrica `⊖` sobre huellas.** Para cada `plantilla` presente en
alguna de las dos: si aparece en ambas con multiplicidades **sintácticamente
iguales**, no entra en el resultado; en cualquier otro caso entra, con las
multiplicidades de ambos lados. `delta = ∅` ⟺ las dos huellas son idénticas
como mapas.

### 1.2 Predicado de distribución (observable, compartido)

Se define **una sola vez** y lo usan tanto §1.3 como §1.4. Depende solo de
datos observables, **nunca de intención**:

| Predicado | Definición |
|---|---|
| `distribución_por_token` | El rank propietario deriva de `batch`, `token` o de la partición de datos, **no** del experto elegido. |
| `distribución_por_experto` | El rank propietario deriva de la asignación de experto sobre `ep` / `efsdp`. |

La **dirección** entre ambas distingue `Dispatch` de `Combine`.

### 1.3 Roles: vocabulario y clasificador por fases

**Vocabulario: 13 roles conocidos + `Opaque` = 14 categorías.**

`ColLinear`, `RowLinear`, `TPReplicatedLinear`, `Attention`, `Norm`,
`Embedding`, `LMHead`, `Router`, `GroupedGEMM`, `Dispatch`, `Combine`,
`LossReduction`, `OptimizerUpdate` · centinela: `Opaque`.

**Se aplica la política de fases de §1.6.4.**

#### 1.3.1 Fase 1 — roles semánticos específicos

| Guarda | Rol |
|---|---|
| Almacenamiento indexado por token de entrada contra una tabla `[V, ·]` | `Embedding` |
| Proyección final a logits de vocabulario | `LMHead` |
| Proyección a puntuaciones por experto que decide encaminamiento | `Router` |
| Producto matricial por experto sobre un parámetro con eje `expert` | `GroupedGEMM` |
| Normalización sobre el eje de rasgos, con parámetro de escala o sin él | `Norm` |
| Producto de puntuaciones QKᵀ, softmax sobre posiciones, o aplicación a V | `Attention` |
| Redistribución de `distribución_por_token` a `distribución_por_experto` (§1.2) | `Dispatch` |
| Redistribución de `distribución_por_experto` a `distribución_por_token` | `Combine` |
| Reducción de la pérdida escalar | `LossReduction` |
| Actualización de parámetros por el optimizador | `OptimizerUpdate` |

#### 1.3.2 Fase 2 — lineal genérico, solo por la componente TP

Si ninguna regla de la fase 1 aplica **y** el operador es lineal:

| Componente TP del parámetro | Rol |
|---|---|
| `tp ↦ Shard(output_feature)` | `ColLinear` |
| `tp ↦ Shard(input_feature)` | `RowLinear` |
| `tp ↦ Replicate` | `TPReplicatedLinear` |

Tres precisiones obligatorias:

1. Se usa el **placement lógico canónico**, anterior a materializaciones
   transitorias como el `AllGather` de FSDP.
2. Se inspecciona la **matriz principal**, no el bias.
3. **`dp_r`, `dp_s` y `efsdp` no intervienen** en esta clasificación.

#### 1.3.3 Fase 3 — residuo

Si el operador no es lineal, o su componente TP no encaja en ninguna fila de la
fase 2 → `Opaque`.

**Por qué fases y no guardas negativas.** Un "todos los anteriores" textual es
correcto hoy y frágil mañana: con fases, un rol nuevo añadido a la fase 1
**preempta automáticamente** al fallback lineal, sin reescribir nada.
`FusedQKVLinear` se resuelve en la fase 2 —parámetro shardeado en
`output_feature`, luego `ColLinear`— sin que su nombre intervenga.

### 1.4 Transiciones: vocabulario y clasificador por fases

**Nueve transiciones semánticas, conjunto cerrado:** `AllReduce`,
`ReduceScatter`, `AllGather`, `AllToAll`, `Broadcast`, `SendRecv`, `Dispatch`,
`Combine`, `NONE`.

**`OpaqueTransition` es un centinela, no una décima transición.** Tolerancia
cero (E6a).

**Firma completamente observable:**

```
(op_comunicacion, distribución_entrada, distribución_salida, grupo)
    → transicion
```

> El término es `op_comunicacion`, no `op_colectiva`: una transferencia punto a
> punto **no es un colectivo**, y con la redacción anterior la guarda «no hay
> colectivo» habría sido cierta también para un `send`/`recv`. Bajo la política
> de fases, dos guardas aplicables en la misma fase producen
> `HUELLA_NO_DERIVABLE`, luego **toda transferencia punto a punto habría sido
> no derivable**. No era un problema de redacción sino de solape lógico.

**Se aplica la política de fases de §1.6.4.**

#### Fase 1 — guardas específicas

| Guarda | Transición |
|---|---|
| **No hay operación de comunicación** | `NONE` |
| Materializa un `Partial(op)` a `Replicate` | `AllReduce` |
| Materializa un `Partial(op)` a `Shard(id)` | `ReduceScatter` |
| Convierte `Shard(id)` en `Replicate` | `AllGather` |
| Replica desde un rank raíz | `Broadcast` |
| **Transferencia lógica punto a punto entre ranks de un mismo grupo** | `SendRecv` |
| Intercambio total, `distribución_por_token → distribución_por_experto` | `Dispatch` |
| Intercambio total, `distribución_por_experto → distribución_por_token` | `Combine` |

**Unidad de conteo de `SendRecv`:**

> Un envío y su recepción correspondientes constituyen **una sola transición
> lógica dirigida** `SendRecv`, observada **antes de cualquier descomposición
> del backend**.

Así una implementación interna de NCCL no modifica el plan lógico —coherente
con §1.8, que deja el plan físico fuera de alcance— y un intercambio
bidireccional cuenta como dos transferencias **solo si transporta dos payloads
dirigidos distintos**.

**Alcance de grupo de `SendRecv`.** La guarda no está restringida a `pp`: si el
paralelismo de contexto es *ring attention* —rotación de KV por envíos punto a
punto sobre `cp`—, una guarda estrecha dejaría esos envíos en
`OpaqueTransition`, que tiene **tolerancia cero** en E6a, y **E0 fallaría de
forma garantizada** sin decir nada sobre la huella. El grupo queda registrado
en `grupo_comunicacion`, que conserva la distinción entre pipeline, `cp` y
cualquier otro eje. El `SendRecv` sobre `pp` no conlleva cambio de placement
porque `pp` parte el grafo; sobre otros grupos es una rotación y lleva
placements con normalidad.

#### Fase 2 — residual cerrado

> Intercambio total que **no satisface ninguna de las dos guardas
> `token → experto` ni `experto → token`** → `AllToAll`.

`Dispatch` y `Combine` tienen así **precedencia semántica**, y `AllToAll` queda
como residual cerrado. Una guarda del tipo «sin cambio de distribución» sería
demasiado estrecha: dejaría fuera intercambios como **secuencia↔cabeza**, que
no son ni dispatch ni combine y sí son intercambios totales legítimos.

#### Fase 3 — residuo

`OpaqueTransition`.

La precedencia entre `AllToAll` y `Dispatch`/`Combine` la decide **el cambio de
eje de distribución según §1.2**, nunca el nombre de la operación: un
`all_to_all` de MoE que cruza de por token a por experto es `Dispatch` aunque
el código lo llame `all_to_all`.

### 1.5 Placements y grupos

**Grupos de comunicación:** un eje simple entre `dp_r`, `dp_s`, `efsdp`, `tp`,
`ep`, `cp`, `pp`, o un producto canónico ordenado de ejes simples:

```
grupo_comunicacion := eje | product(eje_1,...,eje_n)
```

Un producto se serializa en el orden de la malla declarada, sin duplicados, y
solo cuando el HEAD construye una malla unidimensional aplanando esos ejes. No
autoriza a fusionar grupos por conveniencia analítica.

```
Placement :=
    Dense  { eje ∈ malla_densa    ↦ Replicate | Shard(id) | Partial(op) }
  | Sparse { eje ∈ malla_dispersa ↦ Replicate | Shard(id) | Partial(op) }
```

- Cada mapa es **total sobre su propio dominio**, lleva etiqueta
  `Dense`/`Sparse` y se serializa con las claves en orden canónico.
- `Shard` toma un `id_semantico` (§1.6.3), no una expresión.
- `op ∈ {Sum, Max}`.

**`Shard(axis_opaque)` está PROHIBIDO** → `HUELLA_NO_DERIVABLE` (§1.9).

**`pp` no aparece nunca en un placement.** No shardea un tensor: parte el
grafo. Solo aparece como `grupo_comunicacion`, y su única transición es
`SendRecv`.

**Regla de `dp_r`:** `dp_r ↦ Shard(id)` es admisible únicamente cuando

```
clase_tensor ∈ {activation, control_metadata}  ∧  id ∈ {batch, token}
```

DDP replica parámetros pero particiona el lote global, y los índices y pesos de
routing acompañan a esa partición. Para `control_metadata` se exige además que
la partición **proceda de la activación correspondiente sin mezcla entre
ranks**. Para `param`, `grad` y `optimizer_state`, `dp_r` solo admite
`Replicate` o `Partial(op)`.

### 1.6 `firma_tensor`

```
firma_tensor := (forma_normalizada, clase_dtype, clase_tensor)
```

#### 1.6.1 Símbolos: tabla normativa

| Símbolo | Significado |
|---|---|
| `B` | Tamaño lógico de lote |
| `S` | Posiciones de secuencia |
| `D` | Anchura del residual / modelo |
| `F` | Anchura intermedia de FFN |
| `H` | Cabezas de query |
| `Hkv` | Cabezas KV antes de repetición |
| `Dh` | Dimensión por cabeza |
| `E` | Expertos |
| `V` | Vocabulario |
| `L` | Capas |
| `K` | Expertos seleccionados por token (top-k) |
| `C` | Posiciones de capacidad por experto |
| `Qn` | Componente no-posicional por cabeza de Q/K en MLA |
| `Qr` | Componente rotatoria por cabeza de Q/K en MLA |
| `Dv` | Dimensión por cabeza de V en MLA |
| `Rkv` | Rango latente comprimido KV en MLA |

**Literales:**

- Un literal **estructural** —el `2` de QKV fusionado, el `±1` de offsets— es
  un **coeficiente** del polinomio.
- Un literal que representa una **dimensión arquitectónica** debe traducirse a
  su símbolo **con procedencia demostrable**.
- Si no puede hacerse → `HUELLA_NO_DERIVABLE`.

> **Colisión de nombres.** Estos símbolos son los del preregistro, no los del
> repositorio. En el HEAD, `topk_scores_BLK` usa `L` para longitud de
> secuencia; aquí `S` = secuencia y `L` = capas. La traducción es tarea de B, y
> un error de traducción es cuestión de `spec_suficiente`.

#### 1.6.2 Álgebra de expresiones

**Forma normal — mapa, no multiconjunto:**

```
polinomio_normal := mapa
    multiconjunto_de_factores  →  coeficiente entero NO NULO
```

```
E + 1            →  { {E}: 1,  ∅: 1 }
H·Dh + 2·Hkv·Dh  →  { {H,Dh}: 1,  {Hkv,Dh}: 2 }
S − 1            →  { {S}: 1,  ∅: −1 }
```

**Procedimiento:** aplicar las identidades arquitectónicas → distribuir
exhaustivamente → agrupar por multiconjunto de factores sumando coeficientes →
eliminar entradas de coeficiente cero.

**Igualdad = igualdad de mapas.** No se factoriza, no se cancela más allá de
agrupar, no se resuelve ninguna ecuación.

**Positividad:** la evaluación del polinomio bajo la configuración del PE debe
ser **estrictamente positiva**. Admite `S − 1`, excluye extensiones nulas o
negativas.

**Identidades arquitectónicas — sistema de expansión validado:**

```
identidades_arquitectonicas := mapa parcial finito  símbolo → polinomio
```

1. Cada símbolo aparece como lado izquierdo **como máximo una vez**.
2. El grafo `lhs → símbolos(rhs)` es **acíclico**.
3. La sustitución es recursiva, en **orden topológico**.
4. Cada identidad **cita su procedencia arquitectónica**: definición o aserción
   del modelo. **Nunca se infiere porque dos valores coincidan** en el PE.
5. Antes de aplicarla a un par, debe ser **válida bajo la configuración de
   ambos lados**. Si no → `HUELLA_NO_DERIVABLE`.

(1)–(3) garantizan terminación y confluencia. (5) evita el falso negativo
inverso: que una identidad congelada imponga `D = H·Dh` en una revisión
histórica donde ya no se cumplía.

**Validada por PE, nunca universal:** `D → H · Dh` se aplica en `PE_dense`,
donde el Llama debugmodel fija `D=256`, `H=16`, `Dh=16`. No se aplica en
`PE_moe`: DeepSeek V3 MLA desacopla el residual (`D=256`) de las dimensiones
por cabeza Q/K (`128+64`) y V (`128`). El dry-run comprobó además que cambiar a
Qwen3-MoE no restauraría universalidad (`D=256`, `H=16`, `Dh=128`). Forzar la
identidad en cualquiera de esos modelos violaría la exigencia de procedencia y
validez de (4)–(5). Las demás identidades se descubren y congelan en E0.

#### 1.6.3 Eje tensorial

```
eje_tensorial     := (id_semantico, expr)
forma_normalizada := multiconjunto de ejes tensoriales
```

**Vocabulario cerrado (21 + centinela):**

```
batch, seq, token, query_pos, key_pos,
model, input_feature, output_feature,
head, kv_head, head_dim, ffn_hidden, vocab,
expert, topk, capacity, expert_offset, layer,
routed_item, kv_latent, attention_feature,
axis_opaque
```

**`routed_item` (identidad 19, añadida en el dry-run de calibración de E0,
§8.3.5):** una ocurrencia token–experto producida por expansión top-k, antes o
después de la ordenación por experto. Ver §1.6.4.B para su regla de asignación
y §8.3.5 para la procedencia en el código real de referencia.

**Extensión MLA cerrada en E0 (identidades 20–21):** `kv_latent` es el eje
comprimido de rango `Rkv` entre `wkv_a` y `wkv_b`; `attention_feature` es el
aplanado `H·Dv` de la salida por cabezas antes de `wo`. No se identifica con
`model`: en `PE_moe`, `H·Dv=2048` y `D=256`.

**Buena formación:** ningún `id_semantico` **conocido** se repite dentro de una
misma forma. **`axis_opaque` SÍ puede repetirse** y conserva su multiplicidad
— de lo contrario dos ejes desconocidos colapsarían el caso en "ambiguo" y E6d
subestimaría justo la opacidad que mide.

**No ordenado.** El orden de ejes es layout físico, fuera de alcance (§1.8):

```
{(batch,B),(seq,S),(model,·)}  ≠  {(token,B·S),(model,·)}    # fusión SÍ cuenta
{(batch,B),(seq,S),(model,·)}  =  {(seq,S),(batch,B),(model,·)}
```

**La serialización de `forma_normalizada` usa orden canónico de sus elementos,
aunque ese orden no tenga semántica.** Sirve para que dos huellas iguales se
escriban igual; no para distinguirlas.

#### 1.6.4 Asignación de identidad, por fases

```
(clase_tensor, rol, uso del eje)  →  id_semantico
```

**POLÍTICA DE FASES — común a las tres tablas, y a los clasificadores de §1.3
y §1.4:**

> En cada fase:
> - **más de una regla aplicable → `HUELLA_NO_DERIVABLE`**;
> - **exactamente una → devuelve el valor y termina**;
> - **ninguna → continúa a la fase siguiente**.
>
> **Agotadas las fases → el centinela** (`axis_opaque` aquí; `Opaque` en §1.3;
> `OpaqueTransition` en §1.4).
>
> **Las tablas A y C constan de su fase tabular seguida de ese residual.** La
> tabla B tiene una fase intermedia adicional (fase 2).

**Partición base:** `input_feature` / `output_feature` se reservan al
**almacenamiento** de operadores lineales (`param`, `grad`,
`optimizer_state`) — salvo el fallback de la fase 2 de la tabla B.

**A. Almacenamiento** (`param`, `grad`, `optimizer_state`) — fase tabular:

| Rol | Ejes |
|---|---|
| `ColLinear`, `RowLinear`, `TPReplicatedLinear` | `(output_feature, ·)`, `(input_feature, ·)` |
| `Embedding` | `(vocab, V)`, `(output_feature, ·)` |
| `LMHead` | `(vocab, V)`, `(input_feature, ·)` |
| `Router` | `(expert, E)`, `(input_feature, ·)` |
| `GroupedGEMM` | `(expert, E)`, `(output_feature, ·)`, `(input_feature, ·)` |
| `Norm` | `(model, ·)` |

Residual: `axis_opaque`.

**Pesos atados.** El mismo almacenamiento recibe `output_feature` bajo
`Embedding` e `input_feature` bajo `LMHead`; es correcto porque el rol difiere.
**Pero solo si cada plantilla tiene un rol inequívoco.** Si un único colectivo
sirviera a `Embedding` y `LMHead` simultáneamente, la doble interpretación
produce `HUELLA_NO_DERIVABLE`. El dry-run debe comprobar exactamente eso.

**B. `activation` — fase 1, identidades específicas:**

| Uso del eje | Identidad |
|---|---|
| Lote, como eje propio | `batch` |
| Posición, como eje propio, fuera de scores | `seq` |
| Lote y posición aplanados en un solo eje, **sin expansión top-k** | `token` |
| Ocurrencia token–experto tras expansión top-k (dispatch/combine de MoE), como eje propio o aplanado con lote y posición, antes o después de ordenar por experto | `routed_item` |
| Los dos ejes de posición de un tensor de scores/probabilidades bajo `Attention` | `query_pos` y `key_pos` |
| Residual stream | `model` |
| Intermedio de FFN | `ffn_hidden` |
| Cabezas de query | `head` |
| Cabezas KV antes de repetición | `kv_head` |
| Dimensión por cabeza | `head_dim` |
| Rango latente KV comprimido de MLA | `kv_latent` |
| Cabeza y dimensión V aplanadas antes de `wo` en MLA | `attention_feature` |
| Expertos en logits del router | `expert` |
| Ranura top-k de scores o pesos de combinación **diferenciables** | `topk` |
| Vocabulario en logits de salida | `vocab` |
| Apilado de capas | `layer` |

**B — fase 2, fallback del eje lineal fusionado:** si ninguna regla de la fase 1
aplica **y** el eje es la salida de un operador lineal **todavía no
descompuesta** → `output_feature`.

Resuelve el eje QKV fusionado: su extensión `(H + 2·Hkv)·Dh` la expresa el
polinomio, pero el eje no es `head` ni `kv_head` —esas aparecen tras el
`view`/`split`— y sin el fallback caería en `axis_opaque`, con
`HUELLA_NO_DERIVABLE` si `tp` lo shardea. Al descomponerlo, **T4 detecta
correctamente la división**.

> **Extensión MLA de E0:** las proyecciones todavía fusionadas usan este
> fallback. Tras el split, `kv_latent` y `attention_feature` cubren los dos
> ejes propios que el dry-run demostró necesarios; Q/K/V conservan `head_dim`
> con expresiones distintas (`Qn+Qr`, `Qn+Qr`, `Dv`) sin forzar un único `Dh`.

**B — fase 3:** `axis_opaque`.

**C. `control_metadata`** — fase tabular, mismos ejes estructurales que B:

| Uso del eje | Identidad |
|---|---|
| Lote, como eje propio | `batch` |
| Posición, como eje propio | `seq` |
| Lote y posición aplanados | `token` |
| Ranura top-k (índices o asignaciones) | `topk` |
| Experto (counts, máscaras) | `expert` |
| Capacidad por experto | `capacity` |
| Offsets de grouped GEMM | `expert_offset`, extensión `E+1` |

Residual: `axis_opaque`.

En el HEAD los índices top-k existen como `[B, S, K]` **antes** del aplanado:
asumir `(token, ·)` sin más habría fallado.

#### 1.6.5 `clase_dtype` — tipo + anchura

`f32` (fp32/tf32) · `f16` (bf16/fp16) · `f8` · `f4` · `i64` · `i32` ·
`i8` (int8/uint8) · `bool`.

#### 1.6.6 `clase_tensor` y procedencia de gradientes

`param` | `grad` | `activation` | `optimizer_state` | `control_metadata`.

**Regla de procedencia (define qué es `grad`):**

> El cotangente de un **parámetro** es `grad` y hereda las identidades del
> parámetro.
> El cotangente de una **activación** sigue siendo `activation` y hereda las
> identidades de su primal.

Sin esto, el gradiente de activación de un `RowLinear` caería en la tabla A,
que no puede describir `{(batch,B),(seq,S),(model,·)}`. La distinción además
separa el `ReduceScatter` de gradientes de parámetro (FSDP) del `AllReduce` de
gradientes de activación (TP).

**No existe `collective_payload`:** no sería excluyente con las demás, y la
condición de payload ya la expresa `transicion != NONE`.

- Índices top-k, asignaciones, offsets, counts y máscaras → `control_metadata`.
- Logits del router y pesos de combinación **diferenciables** → `activation`.
- **La clase sigue el origen semántico.** Un bucket o buffer aplanado de
  gradientes de parámetro sigue siendo `grad`.

### 1.7 Multiplicidad

Colapso por índice estructural repetido, con multiplicidad simbólica: `L`,
`2·L`, `L_moe`, `2·L_moe`, `L - L_moe`, `P - 1`, `1`.

`2·L` se añadió durante E0 al descomponer el `FeedForward` SwiGLU del PE dense:
`w1` y `w3` tienen la misma plantilla estructural en cada capa (ambos son
`ColLinear` de `D` a `F`) y el mapa de huella debe sumar sus ocurrencias, no
mantener dos claves idénticas con multiplicidad `L`.

La misma razón exige `2·L_moe` para `w1/w3` de los expertos compartidos y
enrutados: en cada capa MoE ambas matrices tienen la misma plantilla dentro de
su familia densa o dispersa, respectivamente.

Un cambio en el **valor** de un símbolo no es cambio de plan. Un cambio en la
**expresión** sí lo es.

### 1.8 Transparencia y relación de cociente

**Una operación es transparente si y solo si se cumplen las cuatro
condiciones:**

1. **T1** — no introduce ninguna transición distinta de `NONE`.
2. **T2** — no provoca redistribución implícita de ningún operando.
3. **T3** — todos los operandos tensoriales relevantes tienen placements
   mutuamente compatibles sin redistribución, y el mapa de placement de salida
   es equivalente al mapa común de entrada bajo T4.
4. **T4** — para cada `eje_malla ↦ Shard(id)` de entrada debe existir en la
   forma de salida **exactamente un** eje con ese mismo `id_semantico`, y su
   `expr` debe ser idéntica. Sin división ni fusión.

**T4 solo puede certificar transparencia para `id ≠ axis_opaque`.**

Las operaciones transparentes no generan plantilla y no rompen la adyacencia
productor→consumidor. **Su efecto no se borra:** todo cambio de forma, clase de
dtype o clase de tensor se propaga a la `firma_tensor` de la plantilla
siguiente. Una permutación no cambia la forma; una fusión de ejes sí.

**Ninguna operación es transparente por su nombre.** Una `Norm` sobre `model`
shardeado viola T1.

**Relación de cociente.** El plan lógico es el cociente de la traza por `~`:

- **Q1 — orden temporal.** **Dos trazas que difieren únicamente en el orden
  temporal de sus sitios son equivalentes.** Válido **únicamente sobre una
  partición de etapas fija**.
- **Q2 — microbatch.** Los sitios que difieren solo en índice de microbatch
  colapsan. Cada arista `pp` cuenta **una vez**, con multiplicidad del grafo de
  etapas (`P-1` en cadena lineal), **nunca** por número de microbatches.
- **Q3 — recompute.** Un sitio cuya única causa es la rematerialización de otro
  ya presente colapsa en él.

**No se cocienta, y por tanto sí cambia el plan:** la partición de etapas, el
conjunto de aristas `pp`, la asignación etapa→módulo. Un schedule entrelazado
sobre la misma partición es el mismo plan; repartir las capas de otra forma es
un plan distinto.

**Fuera de alcance por construcción:** orden de ejes tensoriales (layout), pico
de HBM, métricas temporales, fusión de kernels, algoritmo NCCL, plan físico.

### 1.9 `HUELLA_NO_DERIVABLE`

`AMBIGUOUS` pertenece al gold label de A. **B no puede producirlo.** Cuando B
no puede derivar una huella conforme a §1, registra `spec_suficiente = no` y
`huella = HUELLA_NO_DERIVABLE`. Causas:

- `Shard(axis_opaque)`;
- literal no traducible con procedencia (§1.6.1);
- **más de una regla aplicable dentro de una misma fase** de cualquiera de los
  tres clasificadores (§1.3, §1.4, §1.6.4);
- identidad arquitectónica no validable en algún lado (§1.6.2, cond. 5);
- rol doble en una misma plantilla (pesos atados);
- cierre sin frontera justificable y no escalable.

`diff_huella ∈ {IDÉNTICA, DISTINTA, NO_DERIVABLE}`. `NO_DERIVABLE` **no**
cuenta como FP ni como detección —las fórmulas de §2.3 exigen `= DISTINTA`—
pero se reporta como recuento propio y alimenta E4.

**Cuestión abierta declarada:** la política del gate ante `NO_DERIVABLE`
—fallar abierto o cerrado— **no se decide en el piloto** y debe fijarse antes
de la fase sombra, porque cambia la tasa operacional.

---

## 2. Gold label

Se produce **exclusivamente** desde el diff anotado, el título, la descripción
y la discusión del PR. Nunca desde la huella. Siempre referido a un PE.

### 2.1 Estado por PE

| Estado | Criterio |
|---|---|
| `CHANGE` | El plan lógico pretendido **en ese PE** cambia semánticamente. |
| `REFACTOR` | Ese PE es alcanzado y su plan lógico pretendido permanece idéntico, aunque el diff modifique nombres, estructura o incluso APIs de paralelización. |
| `AMBIGUOUS` | No puede determinarse lo anterior desde la evidencia permitida. |
| `NO_ALCANZADO` | Los cambios del PR no alcanzan ese PE. |

**Ante duda sobre alcanzabilidad se usa `AMBIGUOUS`, nunca `NO_ALCANZADO`.**

### 2.2 Derivación de la etiqueta del PR

```
si algún PE = CHANGE                              → CHANGE
si no, y algún PE = AMBIGUOUS                     → AMBIGUOUS
si no, y algún PE alcanzado (todos REFACTOR)      → REFACTOR
si no (ningún PE alcanzado)                       → FUERA_DE_PE
```

**La unidad estadística es el PR, no la pareja PR×PE.**

El **selector de ficheros de §4.1** y la **lista de APIs de D1 (§4.2)**
seleccionan candidatos; **nunca** asignan `CHANGE` automáticamente.

### 2.3 Fórmulas

```
FP_PE(p,e)   ⟺ estado(p,e) ∈ {REFACTOR, NO_ALCANZADO} ∧ diff_huella(p,e) = DISTINTA
FP_PR(p)     ⟺ ∃e : FP_PE(p,e)
FP_GATE(p)   ⟺ gold_label(p) ∈ {REFACTOR, FUERA_DE_PE} ∧ ∃e : diff_huella(p,e) = DISTINTA
detectado(p) ⟺ ∃e : estado(p,e) = CHANGE ∧ diff_huella(p,e) = DISTINTA
```

En un PR `REFACTOR` todo PE está en `{REFACTOR, NO_ALCANZADO}`, luego `FP_PR` y
`FP_GATE` coinciden. `FP_NO_ALCANZADO` en un PR `CHANGE` por otro PE es **fallo
de localización**: se registra aparte y no cuenta en `FP_GATE`.

---

## 3. Cadencia medida

```
Ventana:                 2026-05-17 → 2026-08-15
PRs mergeados totales:   363   (28,23 / semana)
PRs cualificantes:       137   (10,66 / semana)

N_q                  = 137
Tasa operacional máx = 1/137 = 0,73%   (SLO: ≤1 falsa alarma/trimestre)
```

**Método:** commits de **primer padre** de `main` dentro de la ventana; los 363
terminan en `(#PR)`, proxy limpio de merges/squash. Los 137 son los que tocan
el selector de §4.1. **No ajusta ningún umbral del estudio.**

---

## 4. Población y muestreo

### 4.1 Selector de ficheros cualificantes (literal, INMUTABLE)

```
torchtitan/distributed/**/*.py

torchtitan/models/**/
  model.py | moe.py | parallelize.py | sharding.py |
  expert_parallel.py | token_dispatcher.py | attention.py |
  decoder.py | feed_forward.py | embedding.py | linear.py |
  nn_modules.py | rmsnorm.py | vision_encoder.py |
  vision_encoder_sharding.py | mtp.py | dist_gemm.py | layers.py

torchtitan/models/common/config_utils.py

torchtitan/experiments/**/
  model.py | parallelize.py | pipeline.py | moe_replacement.py |
  hf_sharding.py | module_conversion.py | ep_*.py

torchtitan/protocols/module.py
torchtitan/protocols/sharding.py
torchtitan/overrides/moe_token_dispatcher.py
```

Un selector más grueso (`models/**/*.py` + `distributed/**/*.py`) habría dado
153. Se documenta; **no gobierna**.

### 4.2 Definición operacional de "negativo difícil"

Un PR con `gold_label = REFACTOR` es **difícil** si su diff cumple al menos una
de las tres, en código alcanzable desde algún PE alcanzado:

- **D1** — modifica un punto que construye o muta entradas de plan, malla o
  placements: `parallelize_module`, `DeviceMesh`, `Shard` / `Replicate` /
  `Partial`, `output_layouts`, `use_local_output`, partición de etapas,
  composición de la malla dispersa.
- **D2** — renombra, mueve, reanida o cambia la clase de un módulo que aparece
  como `rol_productor` o `rol_consumidor` en la huella de referencia de ese PE.
- **D3** — reescribe el camino forward de un bloque alcanzable: divide, fusiona
  o reordena operaciones.

**No bastan** (negativos blandos, excluidos): fontanería de configuración,
logging, anotaciones de tipo, cambios solo en tests dentro de esos ficheros,
reformateo.

### 4.3 Muestreo, `par_id` y parada

```
semilla = 20260815

1. Enumerar los 137 PRs cualificantes en orden cronológico → índices 0..136.
2. Permutación fija: random.Random(20260815).shuffle(indices).
3. Asignar un par_id OPACO a cada uno de los 137, en ese momento.
   Permutación y par_id se registran ANTES de etiquetar nada.
4. A etiqueta en ese orden hasta acumular simultáneamente:
       20 × (gold_label = REFACTOR ∧ es_negativo_dificil = sí)
       10 × (gold_label = CHANGE)
5. Se registran: PRs etiquetados hasta el corte, y los recuentos de
   AMBIGUOUS, FUERA_DE_PE y REFACTOR-blando descartados.
```

Los `par_id` se asignan **antes del etiquetado** para que el hash de control de
§5.3 no pueda depender de etiquetas ya vistas.

La parada no puede ocurrir antes del elemento 30, luego los **30 primeros se
etiquetan siempre**: base muestral de §8.2. Si se agotan los 137 sin alcanzar
20 negativos difíciles → E3.

Exclusión previa: PRs que solo tocan docs, tests, CI, logging o dependencias.
**No se sustituye ningún negativo difícil por uno blando.**

### 4.4 Conjunto de B y bases de cada métrica

```
conjunto_B = primario ∪ primeros_30       (deduplicado)

primario    = 20 negativos difíciles + 10 CHANGE
primeros_30 = los 30 primeros de la permutación, sea cual sea su etiqueta
```

| Métrica | Base |
|---|---|
| Principal (`FP_PR`), sensibilidad, E4 | primario (30) |
| Intervalo de tasa operacional | **exclusivamente** `primeros_30` |
| Cobertura (§8.2) | `primeros_30` |
| **E6 (los cuatro)** | **huellas de referencia completas de los dos PEs** |

B recibe el conjunto **mezclado y sin distinción de subconjunto**.

---

## 5. Cegado y protocolo de B

### 5.1 Cegado: dos personas (obligatorio)

1. **A** clasifica y custodia los gold labels. No ve ninguna huella, nunca.
2. **A** entrega a **B** pares de árboles fuente anonimizados junto con la
   definición completa de ambos PEs.
3. **Qué significa "sin diff":** B **puede** generar un diff mecánico entre los
   dos árboles del par — lo necesita para el cierre. Lo que no recibe es **diff
   anotado, mensaje de commit, título, descripción, discusión, ID de PR ni
   indicación de subconjunto**. La prohibición es sobre la intención, no sobre
   la comparación textual.
4. El orden antes/después dentro de cada par se aleatoriza con la misma
   semilla; el mapeo se sella.
5. Los ficheros se unen solo tras el congelado de ambos.

```
gold_labels.csv     # solo A
fingerprints.csv    # solo B
```

### 5.2 Delta pareado con cierre de impacto

**La huella de referencia del HEAD NO es base histórica.** Sirve como
**catálogo de roles** y como calibración de E0 y E6, nada más.

```
C₁, C₂ := cierre de impacto del cambio en cada lado del par

delta_huella(p, PE) := Huella(C₂, lado₂) ⊖ Huella(C₁, lado₁)

diff_huella = IDÉNTICA  ⟺  delta_huella = ∅
```

`⊖` es la diferencia simétrica definida en §1.1.

**Definición comprobable del cierre:**

> El cierre es el **menor punto fijo** que contiene los nodos modificados y se
> propaga por dependencias de datos y de control hasta que, **en cada
> frontera**, coinciden en ambos lados: `rol`, `placement`, `transicion`,
> `firma_tensor`, `grupo_comunicacion` y `multiplicidad`.

Sin correspondencia de frontera justificable → **escalada a `desde_cero`**. El
certificado de frontera se registra.

**Escalada obligatoria** si el cambio afecta a: construcción o composición de
malla · resolución de configuración global · partición `pp` · cualquier
expresión de multiplicidad · un helper compartido alcanzable desde más de un
rol · el alcance de módulos.

**Off-ramp de E5 (decisión antes de etiquetar, nunca después):** si tras E0 la
huella de referencia de un PE cuesta más de **4 horas**, se abandona la unión
de §4.4 y se entrega solo el primario. En ese caso **no se estima tasa
operacional en el piloto**. Se documenta.

### 5.3 Control de anclaje: 20% exacto, doble registro

**Momento del sellado:** después de que A congele `conjunto_B`, y **antes** de
que B reciba ningún árbol. Los `par_id` ya existen desde §4.3.

```
h(par)    = int(SHA256(f"{semilla}|anchor|{par_id}").hexdigest(), 16)
controles = los ceil(0.2 · |conjunto_B|) pares con menor h(par)
```

**Procedimiento en un control:** B deriva la huella **desde cero** sin
consultar la referencia → se **congela y sella con marca de tiempo** → solo
entonces calcula el **delta pareado** → **se guardan los dos resultados**.

```
delta_completo := HuellaCompleta₂ ⊖ HuellaCompleta₁
discrepancia_estructural ⟺ delta_completo ≠ delta_cierre
```

| Observación | Consecuencia |
|---|---|
| Discrepancia en el veredicto `IDÉNTICA`/`DISTINTA` | **El método delta queda invalidado.** Se rehace todo desde cero, o se activa E5. |
| Mismo veredicto, `delta_completo ≠ delta_cierre` | Se reporta por separado; esas huellas **no** se usan para estadísticas estructurales sin advertencia explícita. |

---

## 6. Métricas

**Principal (bloqueante):** `FP_PR` sobre los 20 negativos difíciles.
Criterio: **0 de 20.** Causa raíz: **bug de especificación** (reparable, quema
la muestra, §9) o **inestabilidad intrínseca** (E1).

**Secundaria (no bloqueante):** `detectado(p)` sobre los 10 positivos, con
coincidencia de PE. Objetivo ≥ 8/10.

**Intervalo de tasa operacional**, sobre `primeros_30`:

```
mín = |{p : gold ∈ {REFACTOR, FUERA_DE_PE} ∧ ∃e diff = DISTINTA}| / 30
máx = mín + |{p : gold = AMBIGUOUS ∧ ∃e diff = DISTINTA}| / 30
```

Orientativo, reportado junto al recuento de `NO_DERIVABLE`. La medición válida
del SLO es la sombra de 409 PRs.

**Cobertura:** los dos estimadores de §8.2.

La precisión pesa mucho más que el recall.

---

## 7. Condiciones de abandono

| Cód. | Condición |
|---|---|
| E0 | Algún PE resulta inválido por §1.0, o el dry-run de §8.3 no se cierra antes de firmar. |
| E1 | Un `FP_PR` con causa raíz de inestabilidad intrínseca. |
| E2 | Se necesitan dos revisiones de la especificación **después de la firma**. |
| E3 | Menos de 20 negativos difíciles en los 137 cualificantes. |
| E4 | **≥ 7 de los 30 PRs primarios** con `spec_suficiente_PR = no`. |
| E5 | Se agota el timebox de 5 días (§5.2, §5.3). |
| E6 | Cualquiera de los cuatro sub-umbrales de §8.1. |

**Agregación de E4:**

```
spec_suficiente_PR(p) = no  ⟺  ∃ PE, ∃ lado : spec_suficiente = no
```

---

## 8. Umbrales, cobertura y validación previa

### 8.1 Umbrales de centinela (E6) — RATIFICADOS

Suficiencia de ingeniería, **no estimaciones estadísticas**. **Base: las
huellas de referencia completas de los dos PEs**, por configuración y **nunca
agrupadas**; sobre plantillas normalizadas, **sin ponderar por
multiplicidad**. Los contadores por PR se registran pero **no** gatillan E6.

| Sub | Umbral | Razón |
|---|---|---|
| E6a | **≥1 `OpaqueTransition`** | Nueve transiciones en conjunto cerrado. Una sola sin clasificar significa vocabulario mal escrito. Tolerancia cero. |
| E6b | **> 10%** de plantillas con `Opaque` en **ambos** extremos | "Algo → colectivo → algo" no dice nada. |
| E6c | **> 25%** de `roles_opaque / roles_totales`, con `roles_totales = 2 × plantillas_totales` | Por encima, lo medido es la estabilidad de un vocabulario vacío. |
| E6d | **Se detecta un placement que requeriría `Shard(axis_opaque)`** —imposible dentro de una huella válida porque §1.5 lo rechaza, luego es un **fallo de validación previo**— **OR** `axis_opaque / ejes_totales > 10%` | El vocabulario de ejes es más pequeño y predecible que el de roles. |

**`ejes_totales`** cuenta ocurrencias en la `firma_tensor` de cada plantilla
normalizada de referencia, **una vez por plantilla y sin multiplicidad**. Las
referencias desde `Shard(id)` **no** vuelven a contar el eje.

### 8.2 Cobertura — regla de decisión, NO de abandono

Base muestral: los **30 primeros de la permutación**.

```
cobertura_confirmada = PRs con algún PE en {CHANGE, REFACTOR} / 30
cobertura_posible    = 1 − FUERA_DE_PE / 30
```

| Resultado | Lectura |
|---|---|
| `cobertura_posible < 0,40` | Hacen falta más PEs. |
| `cobertura_confirmada ≥ 0,40` | Cobertura suficiente para continuar. |
| Entre ambas | Indeterminado: ampliar la muestra de cobertura. |

**Añadir un PE tras observar esto activa §9.** No se reutilizan las huellas.

### 8.3 E0: validación del PE y dry-run de calibración

1. La validación usa el **manifiesto definitivo** y ocurre **antes** de
   etiquetar ningún PR.
2. Comprueba **grupos reales, incluido `efsdp`**, no solo grados declarados.
   **Atención al eje `cp` de `PE_dense`:** su cobertura depende de cómo esté
   implementado el paralelismo de contexto, y es el caso que motiva la guarda
   ancha de `SendRecv` en §1.4.

   **Evidencia parcial (calibración, no cierre de E0):**
   `scripts/e0_mesh_validation.py` lanza el `world_size` real declarado (32
   para `PE_dense`, 8 para `PE_moe`) como procesos CPU locales con backend
   `gloo`, importa sin modificar `torchtitan.distributed.parallel_dims` del
   HEAD de referencia, llama a su `build_mesh()` real y ejecuta un
   `all_reduce` real sobre cada malla unidimensional resultante. **Esto NO es
   NCCL/GPU real** — confirma que el código de referencia forma los grupos de
   comunicación declarados y que las colectivas corren, no valida
   rendimiento, ancho de banda ni comportamiento sobre interconexión física
   real.

   Resultado en el HEAD `9a711521ac2973fe230a3f38efc6aedfc7d1f9c6`:

   - `PE_moe` (world size 8): ejes activos con colectiva confirmada — `pp`(2),
     `batch`(2), `loss`(2), `tp`(2), `ep`(2), `efsdp`(2), `fsdp`(2). Coincide
     exactamente con `dp_s`, `tp`, `ep`, `efsdp`, `pp` de la tabla de grados
     (§1.0); `cp` y `dp_r` están en `⊥` y correctamente ausentes.
   - `PE_dense` (world size 32): ejes activos con colectiva confirmada —
     `pp`(2), `batch`(4), `loss`(8), `dp_replicate`(2), `cp`(2), `tp`(2),
     `fsdp`(4). Coincide con `dp_r`, `cp`, `tp`, `pp` de la tabla; `ep` está en
     `⊥` y correctamente ausente.

   **Resuelve la atención al eje `cp`:** bajo `spmd_backend="default"`, `cp`
   **sí** forma su propia malla unidimensional con grupo de comunicación real
   (`dataloading_mesh["cp"]`), independiente de que además participe,
   fusionado con `dp_shard`, en la malla `fsdp` usada para el sharding de
   parámetros. `cp` **está cubierto** como eje propio en `PE_dense`, con
   independencia del backend SPMD elegido — no depende de una decisión de
   implementación que pudiera dejarlo sin grupo propio.

   **Comprobación mecánica HSDP adicional (calibración, no cierre):**
   `scripts/e0_hsdp_trace.py` ejecuta FSDP2 real sobre cuatro procesos CPU/Gloo
   en una malla `dp_replicate=2 × fsdp=2`, con forward, backward y paso de
   optimizador. El profiler confirmó en **los cuatro ranks** `all_gather`,
   `reduce_scatter` y `all_reduce` (`c10d::_allgather_base_`,
   `c10d::_reduce_scatter_base_`, `c10d::allreduce_`). Esto confirma que el
   candidato HSDP de `PE_dense` necesita una transición `AllReduce` sobre
   `dp_r`, además de las transiciones FSDP sobre `dp_s`. No determina todavía
   roles, firmas tensoriales ni multiplicidades completas, y no es NCCL/GPU.

   **Corrección del grupo FSDP dense:** el backend fijado construye
   `fsdp=dp_shard·cp`; en `PE_dense` la colectiva observable usa por tanto
   `product(dp_s,cp)` de tamaño 4, no `dp_s` aislado. En `PE_moe`, `cp=1` y el
   grupo dense se reduce canónicamente a `dp_s`; los expertos routed usan
   `efsdp`. Esta observación motiva la gramática de grupos producto de §1.5.

   **Primera composición completa de siete campos:** para cada una de las nueve
   familias lógicas de parámetros dense se emiten AllGather de parámetro
   (`OptimizerUpdate → rol del operador`), ReduceScatter de gradiente
   (`rol → OptimizerUpdate`) sobre `product(dp_s,cp)` y AllReduce HSDP de
   gradiente sobre `dp_r`: 27 plantillas candidatas. El sharding FSDP usa la
   primera identidad semántica canónica de la forma del parámetro sobre `dp_s`
   y `cp`; el componente TP se conserva. Siguen siendo candidatas hasta cerrar
   la huella completa y el cruce de rutas runtime.

   **Descomposición semántica de almacenamiento (calibración, no cierre):**
   el extractor cataloga por separado las familias lógicas de parámetros dense
   y MoE y emite para cada una la firma §1.6 completa: forma normalizada,
   `clase_dtype` congelada por el manifiesto y `clase_tensor=param`. Una
   validación mecánica rechaza identidades desconocidas o conocidas repetidas,
   expresiones vacías y clases fuera del vocabulario. Esto cierra las firmas de
   almacenamiento, pero todavía no las compone con productor, consumidor,
   placements y transición para formar las siete-tuplas de §1.1.

   Las firmas lógicas de gradiente se derivan uno-a-uno de esas familias:
   conservan forma, rol y multiplicidad, cambian a `clase_tensor=grad` y usan
   exclusivamente `dtype_classes.grad_reduce` del manifiesto. Esta derivación
   no presupone placements de entrada/salida de FSDP/HSDP; dichos placements se
   fijan al componer las transiciones completas.

   El optimizador seleccionado por ambos registros es `default_adamw`, cuya
   configuración revisada fija AdamW fusionado y `amsgrad=false`. Un probe real
   de PyTorch materializa un paso para parámetros fp16 y bf16: `exp_avg` y
   `exp_avg_sq` conservan forma y dtype del parámetro, mientras `step` es un
   escalar fp32. Por cada familia lógica se catalogan por tanto dos estados con
   la forma del parámetro y un estado escalar vacío; todos llevan
   `clase_tensor=optimizer_state` y rol `OptimizerUpdate`. El probe no valida
   todavía sus placements distribuidos.

   Para `PE_moe` el manifiesto congela además
   `moe_comm_backend=standard`, que selecciona `AllToAllTokenDispatcher`. Sus
   IDs top-k, mapa booleano de routing, mapeo token↔routed-item, counts por
   experto y permutación posterior se catalogan como `control_metadata` con
   multiplicidad `L_moe`: IDs/counts `i64`, mapa `bool`. Los IDs conservan
   `[B,S,K]` antes del aplanado; solo los buffers posteriores usan
   `routed_item=B*S*K`. Las puntuaciones completas/top-k/reordenadas se
   catalogan aparte como activaciones `f32`; los payloads expertos y la salida
   combinada usan la clase de baja precisión del manifiesto.

   Para `PE_dense` se catalogan además las activaciones no internas de atención:
   embedding, normas, salida de la frontera Attention, proyecciones/producto
   SwiGLU, salida rowwise, norma final y logits LMHead. `w1/w3` conservan la
   multiplicidad calibrada `2·L`. QKV, separación por cabezas, posiciones y
   tensores internos de atención quedan fuera hasta completar su descomposición
   específica; no se sustituyen por formas residuales aproximadas.

   El manifiesto dense congela `attn_backend=flex` y QKV fusionado. La salida
   lineal previa al split usa el fallback normativo
   `output_feature=(H+2·Hkv)·Dh`; después se separan query `[B,S,H,Dh]` y
   key/value `[B,S,Hkv,Dh]`, estas últimas con multiplicidad `2·L`. También se
   catalogan la salida de atención por cabezas y su forma residual aplanada.
   No se inventa un tensor de scores materializado: FlexAttention puede
   fusionarlo internamente y esa materialización física está fuera de alcance.

   **Resolución del hallazgo MLA dentro de la excepción pre-firma de §9:** se
   añaden los símbolos `Qn`, `Qr`, `Dv`, `Rkv` y solo dos identidades:
   `kv_latent` y `attention_feature`. Q/K/V comparten la identidad semántica
   `head_dim`, pero con expresiones `Qn+Qr`, `Qn+Qr` y `Dv`; no se fuerza `Dh`.
   El latente tras `wkv_a` usa `(kv_latent,Rkv)` y la salida `[B,S,H,Dv]`
   aplanada antes de `wo` usa `(attention_feature,H·Dv)`. En `PE_moe` se
   congelan `Qn=128`, `Qr=64`, `Dv=128`, `Rkv=512` con procedencia directa del
   `_debugmodel`. El audit pasa sin `axis_opaque`; siguen prohibidas las
   asignaciones por coincidencia numérica.
3. **Se derivan las huellas de referencia COMPLETAS de ambos PEs** y se
   calculan allí los cuatro sub-umbrales de E6. Los casos especiales de abajo
   **no las sustituyen**.
4. **Restricción de alcance de E0 (cierra el hueco de §9):** E0 opera
   **exclusivamente** sobre el HEAD de referencia y los casos de calibración.
   **No puede derivar la huella de ningún PR de la población de 137.**

   **Los casos de calibración son configuraciones del mismo HEAD o fixtures
   sintéticos preregistrados. Nunca revisiones, parches ni artefactos derivados
   de los 137 PRs.**
5. **Dry-run de calibración obligatorio**, cerrado antes de firmar:

   - **QKV fusionado con GQA** — extensión `(H + 2·Hkv)·Dh`. **Primer punto de
     tensión: la identidad del eje fusionado.** Verifica el fallback de la fase
     2 de §1.6.4.B y que T4 detecta la división al descomponer.
   - **MoE dispatch/combine, antes y después del aplanado** — índices top-k en
     `[B,S,K]`, offsets `E+1`, counts, máscara de capacidad, `topk_scores`
     diferenciable como `activation`. **Y el estado posterior:**
     `[B,S,K] → [B·S·K] →` buffers ordenados por experto
     (`topk_scores_experts_sorted`, `routed_scores`).

     Ese eje único combina token y ranura top-k.

     ```
     routed_item := una ocurrencia token–experto producida por expansión
                    top-k, antes o después de la ordenación por experto
     ```

     **Decisión (cerrada en el dry-run sintético de E0):** se añade
     `routed_item` como identidad 19. `token` con extensión `B·S·K` se
     descarta como representación de este eje porque no ofrece una
     representación fiel: pierde justo la distinción token/ranura-top-k que
     T4 necesita para rastrear Dispatch/Combine.

     **Procedencia en el código real de referencia** (HEAD
     `9a711521ac2973fe230a3f38efc6aedfc7d1f9c6`, `torchtitan`): en
     `moe.py`/`token_dispatcher.py` el propio código distingue `T` (tokens,
     `B·S` aplanado) de `N` (tokens enrutados, `T·K`) como conceptos
     distintos, con buffers propios (`routed_input_RD`,
     `token_indices_experts_sorted_N`, `topk_scores_experts_sorted_N`). Tras
     el dispatch, la propiedad del dato sigue la distribución por experto, no
     la distribución por token.

     **A diferencia de un rol sin usar, una identidad adicional puede
     solaparse con otra y afectar a T4**: los roles son baratos porque las
     fases los mantienen disjuntos; las identidades no. Verificado: `token`
     (fase 1, sin expansión top-k) y `routed_item` (expansión top-k) son
     mutuamente excluyentes por construcción — ningún eje puede cumplir ambas
     condiciones a la vez, así que no hay ambigüedad de fase.
   - **Embedding y LMHead, atados y sin atar** — partición
     `input_feature`/`output_feature`, y comprobar **si algún colectivo sirve a
     ambos roles** (→ `HUELLA_NO_DERIVABLE`).

   Cada caso produce: identidades usadas, `axis_opaque` generados, identidades
   arquitectónicas necesarias **con su procedencia**. **La suficiencia de las
   21 identidades (18 originales + `routed_item` + 2 de MLA) y de los 13 roles la decide
   este dry-run.**
6. `hash_manifiesto` y el hash del preregistro se recalculan **después** del
   último ajuste de calibración.

---

## 9. Regla de revisión, y su excepción para E0

**Excepción de calibración.** Los resultados de calibración preregistrados de
E0 quedan **excluidos** de esta regla. E0 **puede modificar §1** antes de la
firma y del hash final; esos ajustes son calibración legítima y se documentan.
La excepción es segura porque §8.3.4 prohíbe a E0 tocar cualquier PR de la
población y restringe los fixtures al propio HEAD.

**Entrada en vigor.** Esta regla rige **al cerrarse E0 y firmarse el
preregistro**. Desde ese momento:

> Ver **cualquier huella de un PR de la población** y después modificar §1
> invalida la muestra completa.

Continuar exige un lote nuevo y disjunto, de una ventana temporal distinta, con
semilla nueva registrada. Máximo una revisión posterior a la firma dentro del
timebox. Una segunda revisión necesaria = E2.

Sin esta excepción, `routed_item` no podría ser candidato del dry-run: las
huellas de referencia de E0 también son huellas.

---

## 10. Fichas

### 10.1 `gold_labels.csv` — solo A

```
par_id_opaco:
estado_PE_dense:         CHANGE | REFACTOR | AMBIGUOUS | NO_ALCANZADO
estado_PE_moe:           CHANGE | REFACTOR | AMBIGUOUS | NO_ALCANZADO
gold_label:              (derivado por §2.2)
en_primario:             sí | no
en_primeros_30:          sí | no
es_negativo_dificil:     sí | no
criterio_dificultad:     D1 | D2 | D3 | n/a
pe_que_lo_hace_dificil:  PE_dense | PE_moe | ambos | n/a
ficheros_cualificantes:
justificacion_breve:
```

### 10.2 `fingerprints.csv` — solo B

Una fila por `(par, PE)`.

```
par_id_opaco:
pe:                            PE_dense | PE_moe
es_control:                    sí | no                  # §5.3
escalado_desde_cero:           sí | no + causa          # §5.2

# --- vía delta pareado ---
cierre_C1 / cierre_C2:         descripción
certificado_frontera:          nodos de frontera + coincidencia de los seis campos
identidades_validadas:         lista + procedencia + validez en AMBOS lados
huella_C1 / huella_C2:         mapa plantilla → multiplicidad
                               # plantilla = los siete campos de §1.1
                               # place_*   = Dense{...} | Sparse{...}
                               # firma     = (multiconjunto de (id, expr), clase_dtype, clase_tensor)
delta_cierre:
veredicto_delta:               IDÉNTICA | DISTINTA | NO_DERIVABLE

# --- vía desde cero (controles y escaladas) ---
huella_completa_lado_1 / lado_2:
delta_completo:
veredicto_desde_cero:          IDÉNTICA | DISTINTA | NO_DERIVABLE
sello_desde_cero:              timestamp   # congelado ANTES del delta

# --- comparación de métodos (solo controles) ---
discrepancia_veredicto:        sí | no
discrepancia_estructural:      sí | no   (delta_completo ≠ delta_cierre)

# --- contadores ---
# En pares derivados SOLO por cierre NO son totales: se registran como
# *_cierre y NO alimentan E6, que se calcula sobre las referencias completas.
roles_opaque_cierre / plantillas_totales_cierre /
plantillas_con_opaque_cierre / plantillas_doble_opaque_cierre /
axis_opaque_cierre / ejes_totales_cierre
# En pares completos se registran además los totales.

spec_suficiente_lado_1 / lado_2:   sí | no
regla_no_cubierta:
tiempo_de_aplicacion:              minutos
```

### 10.3 Definición operacional de `spec_suficiente`

> `sí` únicamente cuando **los siete campos de §1.1**, la transparencia y las
> multiplicidades pueden asignarse **citando una regla expresa de §1**, sin
> analogías, sin crear categorías nuevas y sin recurrir a información sobre la
> intención del PR.

Asignar `Opaque`, `OpaqueTransition` o `axis_opaque` **siguiendo una regla
expresa** —incluida la fase de residuo de un clasificador— es
`spec_suficiente = sí`. Las causas enumeradas en §1.9 son `no`.

---

## 11. Salidas del estudio

1. E0: cobertura de cada eje en ambos PEs **con atención expresa a `cp`**,
   huellas de referencia completas, los cuatro E6 calculados sobre ellas,
   resultado del dry-run con las identidades arquitectónicas congeladas y su
   procedencia, la decisión sobre `routed_item`, ajustes de calibración, hashes
   finales, coste real de cada huella de referencia.
2. `N_q`, intervalo de tasa operacional, recuento de `NO_DERIVABLE`, los dos
   estimadores de cobertura, y el dimensionado de sombra de §0.2.
3. Los dos PEs congelados, con `hash_manifiesto`, dominios de malla y
   `particion_pp`.
4. Permutación, semilla, `par_id`, PRs etiquetados hasta el corte,
   `|conjunto_B|`, y si se activó el off-ramp de §5.2.
5. Las fichas, en sus dos ficheros.
6. Recuentos: `FP_PR`, `FP_GATE`, `FP_NO_ALCANZADO`, positivos detectados con
   coincidencia de PE, ambiguos, fuera de PE, los cuatro centinelas **sobre las
   referencias**, `spec_suficiente_PR = no`.
7. Control de anclaje: número de controles, discrepancias de veredicto,
   discrepancias estructurales, y si el método delta quedó invalidado.
8. Pares escalados a `desde_cero` y su causa.
9. Análisis de causa raíz de cada FP.
10. Decisión: **continuar** (redactar issue) o **archivar**, con código de
    abandono si aplica.

El issue no se redacta hasta tener el punto 10.

---

**Firma A:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
**Firma B:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
**Hash del commit de este fichero:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
