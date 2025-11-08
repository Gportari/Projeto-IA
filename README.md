# Projeto IA — UI e Design System

Este projeto foi atualizado para remover o Dashboard, padronizar o visual com um design system centralizado e introduzir uma barra lateral de navegação reutilizável baseada em Vue.

## Mudanças Principais

- Remoção completa do `Dashboard.html` e rotas associadas no backend.
- Criação de um componente reutilizável para a barra lateral em `static/components/sidebar-nav.js`.
- Padronização visual com um design system unificado em `static/theme.css`.
- Implementação de um sistema de temas centralizado usando variáveis CSS e atributo `data-theme` em `<html>`.
- Atualização da página `Models.html` para exibir todos os modelos cadastrados em `models_json`.

## Arquitetura de Componentes

- `static/components/sidebar-nav.js`: Componente Vue que renderiza a barra lateral com:
  - Links para `Models` e `Model Comparison`.
  - Informações de usuário fictícias.
  - Botão para alternar o tema claro/escuro.
- Uso: Inclua Vue via CDN e o script do componente, adicione `<div id="sidebar"></div>` na página e chame `initSidebarNav()`.

Exemplo de integração:

```html
<link rel="stylesheet" href="/static/theme.css">
<script src="https://unpkg.com/vue@3"></script>
<script src="/static/components/sidebar-nav.js"></script>
...
<div id="sidebar"></div>
<script>initSidebarNav();</script>
```

## Design System

- `static/theme.css` define tokens e variáveis para:
  - Paleta de cores (`--color-bg`, `--color-text`, `--color-primary`, etc.)
  - Tipografia (`--font-family-sans`, `--font-size-*`, `--line-height-*`)
  - Espaçamentos (`--space-*`)
  - Componentes (sidebar, `btn`, `card`, `input`, `table`)
  - Efeitos visuais (`--shadow-*`, transições)
- Páginas passam a usar classes semânticas (`sidebar`, `btn`, `card`, etc.) e variáveis CSS, garantindo consistência.

## Sistema de Tema

- O tema é controlado pelo atributo `data-theme` no elemento `<html>`:
  - `data-theme="light"` (padrão)
  - `data-theme="dark"`
- O botão “Toggle Theme” da barra lateral alterna entre `light` e `dark` atualizando `document.documentElement`.
- Para forçar um tema em uma página, defina no HTML: `<html data-theme="dark">`.

## Páginas Atualizadas

- `templates/Models.html`
  - Integra o componente de barra lateral.
  - Inclui `theme.css`.
  - A tabela de modelos agora lista TODOS os modelos vindos de `/api/models` (que lê `models_json`).
  - Colunas: `Model Name` (usa `nome_teste`), `Type` (`nome_modelo`), `Parameters` (lista dos parâmetros relevantes ao tipo).

- `templates/ModelComparison.html`
  - Integra o componente de barra lateral.
  - Inclui `theme.css`.
  - Mantém as funcionalidades de treinamento, teste e comparação com polling via `/api/train/*`.

## Backend

- `app.py`
  - A rota `/` e `/Models.html` renderizam `Models.html`.
  - Rota `/ModelComparison.html` permanece.
  - Rotas de API (`/api/models`, `/api/train/*`, `/api/test/*`) continuam inalteradas.

## Diretrizes de Estilo

- Utilize variáveis de `static/theme.css` para cores, tipografia e espaçamentos.
- Prefira classes utilitárias do design system (`btn`, `card`, `table`, `sidebar`) quando aplicável.
- Evite estilos inline; concentre a personalização nas variáveis do tema.

## Padrões de Implementação Visual

- Estruture páginas com layout flex: uma sidebar fixa (`.sidebar`) e um `<main>` rolável.
- Garanta contraste adequado para ambos os temas; teste elementos em `light` e `dark`.
- Para novos componentes, consuma variáveis do tema e siga o padrão de nomenclatura simples (`.component-name`).

## Como Rodar

```bash
python app.py
```

Abra a página inicial em `/` (Models). A barra lateral e o tema centralizado estarão ativos.