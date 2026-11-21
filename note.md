# LLM-proxies — notes pour préparer un entretien

## Présenter le projet en 30 secondes

« L’objectif de LLM-proxies est de construire une passerelle commune vers plusieurs API de modèles de langage avec LiteLLM. Le projet explore les appels SDK, la communication client–proxy, la configuration YAML, la gestion des clés et des processus, puis le load balancing, les fallbacks et l’observation. LangChain illustre la composition d’un workflow de résumé. Je distingue les comportements vérifiés des intégrations qui restent à valider. »

## Architecture et communication

```text
Client Python → Proxy LiteLLM → API OpenAI / Groq / Cohere
              ← réponse     ← réponse
                      ↓
               Traces Langfuse
```

- **Client** : prépare les messages et envoie un `POST /v1/chat/completions` en JSON.
- **Proxy** : authentifie le client, résout l’alias et appelle le fournisseur avec sa clé API.
- **Fournisseur** : héberge le modèle et génère la réponse.
- **Langfuse** : reçoit les traces ; il ne génère pas de texte.
- Une URL commune et des alias réduisent le code propre à chaque fournisseur. Les capacités des modèles restent différentes.
- Les appels sont synchrones ; `stream=True` permet une réception progressive lorsque le modèle le prend en charge.

## Sections : objectif, techniques et avantages

| Section | Objectif et techniques | Avantage à expliquer |
|---|---|---|
| SDK | Appels directs avec `litellm.completion`, préfixes fournisseur et messages. | Une interface commune sans serveur proxy à démarrer. |
| Paramètres | `temperature`, limites de tokens, streaming et timeout selon le modèle. | Ajuster la génération, la longueur de sortie et l’attente. |
| Proxy simple | API compatible OpenAI, alias YAML, `base_url`, authentification Bearer. | Centraliser les fournisseurs et les accès derrière une seule API. |
| Multi-fournisseurs | Un alias distinct par modèle OpenAI, Groq ou Cohere. | Choisir le fournisseur en changeant l’alias demandé. |
| Environnement | `.venv`, `.env`, `python-dotenv`, vérification des clés et chemins. | Isoler les dépendances et séparer secrets et code. |
| Arrière-plan | `Popen`, suivi des processus, logs et vérification HTTP ; `nohup` en alternative. | Garder le notebook disponible et contrôler le serveur. |
| Load balancing | Alias partagé `global-llm`, `simple-shuffle`, capacités `rpm`. | Répartir les appels entre les déploiements disponibles. |
| Résilience | Retries, timeouts, fallback entre alias, seuil d’échecs et cooldown. | Essayer un secours et éviter temporairement un déploiement en difficulté. |
| Logs | Callbacks succès/échec, JSONL et sortie du processus serveur. | Diagnostiquer les appels et les erreurs à la bonne couche. |
| Observation | `langfuse_otel`, OpenTelemetry, clés et host Langfuse. | Inspecter les durées, modèles, usages et erreurs exportés. |
| LangChain | Loader, prompt, `ChatOpenAI`, parseur et composition avec `\|`. | Assembler un workflow tout en laissant le routage à LiteLLM. |
| Protection | Clé maître dans le projet ; budgets, limites et restrictions à prévoir pour un service partagé. | Distinguer authentification, protection des accès et maîtrise des coûts. |

## Quatre mécanismes à distinguer

- **Load balancing** : répartir les appels normaux entre déploiements.
- **Retry** : refaire une tentative après une erreur considérée comme réessayable.
- **Fallback** : essayer un modèle de secours après un échec éligible du principal.
- **Cooldown** : retirer temporairement un déploiement après trop d’échecs.

Une réponse incorrecte sur le fond ne déclenche pas automatiquement un fallback. Le cooldown du routeur ne déploie pas une fonction serverless et ne pilote pas son cycle de vie.

## Configuration, clés et ports

| Configuration | Usage | Repère |
|---|---|---|
| `llm.yaml` | Appel simple OpenAI | Port 4000, alias `openai-gpt4o-mini` |
| `llm-lb.yaml` | Répartition entre OpenAI et Groq | Port 4001, alias `global-llm` |
| `llm-fallback.yaml` | Secours | `openai-gpt4o-mini` → `groq-mixtral` |
| `llm-lanfuse.yaml` | Observation Langfuse | Port 4002, client `observation_client` |

- `--config` choisit **un YAML** ; le proxy ne fusionne pas tous les fichiers.
- `model_name` est l’alias public ; `litellm_params.model` indique le modèle réel.
- Les anciens alias `groq-gemma9b` et `groq-mixtral` correspondent désormais à GPT-OSS 20B et 120B via Groq : lire le YAML.
- `.env` est un fichier de secrets ; `.venv/` est le dossier de l’environnement Python.
- `LITELLM_MASTER_KEY` authentifie le client auprès du proxy.
- Les clés OpenAI/Groq/Cohere authentifient les appels aux fournisseurs.
- Langfuse nécessite `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` et `LANGFUSE_HOST` du bon projet et de la bonne région.
- Après modification de `.env`, recharger les variables et redémarrer le proxy.

## Points techniques à savoir expliquer

- `Popen` lance un **processus**, pas un thread. `poll()` renvoie `None` pendant son exécution ; `terminate()` demande l’arrêt ; `wait()` attend sa fin.
- `127.0.0.1` désigne la machine exécutant le client : ordinateur local ou runtime distant Colab.
- `Popen` et `nohup` ne survivent pas à la suppression du runtime Colab.
- `GET /v1/models` vérifie le proxy et ses alias, pas la génération chez chaque fournisseur.
- Les callbacks du SDK concernent le processus du notebook ; ils ne sont pas automatiquement partagés avec le proxy.
- JSONL contient un objet JSON par ligne. L’observation utilise `logs/observation-proxy.log`.
- LangChain orchestre **documents → prompt → modèle via LiteLLM → texte** ; LiteLLM route l’appel.

## Questions fréquentes

**Pourquoi un proxy ?** Il centralise configuration, fournisseurs, accès, routage et traces. Il ajoute aussi un service à maintenir et un point de panne.

**Comment savoir quel modèle a répondu ?** `response.model` peut conserver l’alias. Inspecter les en-têtes `x-litellm-model-name` et `x-litellm-model-id` lorsqu’ils sont exposés, puis les logs.

**Pourquoi le même modèle répond-il plusieurs fois ?** `simple-shuffle` ne garantit pas une alternance stricte. Faire de nouveaux appels et compter les déploiements ; réafficher une ancienne réponse ne teste pas le routage.

**Comment protéger le budget ?** Prévoir des clés par application, budgets, limites de débit, restrictions de modèles et suivi fournisseur. La clé maître et `rpm` ne garantissent pas un plafond financier. Certains contrôles LiteLLM nécessitent une base de données.

**Comment valider le projet ?** Tester chaque fournisseur, mesurer la répartition, provoquer un fallback contrôlé et retrouver une trace Langfuse. Une réponse du modèle ne prouve pas que l’export de traces fonctionne.

## Erreurs concrètes à raconter

| Symptôme | Diagnostic / correction |
|---|---|
| `Connection refused` | Vérifier processus, URL, port et journal de la section. |
| `No connected db` | Une clé arbitraire peut être traitée comme clé virtuelle ; vérifier la clé maître. |
| `DEBUG=release` refusé | Normaliser les booléens dans l’environnement du processus proxy. |
| HTTP 401 Groq | Vérifier la clé fournisseur, recharger `.env` et redémarrer. |
| Ancien modèle retiré | Vérifier le catalogue puis mettre à jour le modèle réel dans le YAML. |
| Clés Langfuse absentes | Renseigner les clés du projet Langfuse ; une clé fournisseur ne les remplace pas. |
| Conflit de port | Séparer proxy simple, load balancing et observation sur 4000, 4001 et 4002. |

**En entretien : présenter les techniques et leur utilité, puis préciser les limites.** Des appels OpenAI et Groq ont été observés ; la distribution complète, les fallbacks et l’export Langfuse restent à valider séparément. Cohere et LangChain sont des extensions à configurer et tester.
