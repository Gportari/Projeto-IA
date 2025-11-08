// Sidebar Navigation as a reusable Vue component
// Usage: include Vue 3 CDN and this script, then place <div id="sidebar"></div> in pages and call initSidebarNav()

window.initSidebarNav = function initSidebarNav(options = {}) {
  const { activePath = window.location.pathname } = options;
  const App = Vue.createApp({
    data() {
      return {
        links: [
          { label: 'Models', icon: 'neurology', href: '/Models.html' },
          { label: 'Model Comparison', icon: 'compare', href: '/ModelComparison.html' }
        ],
        user: { name: 'ML Workspace', email: 'user@example.com' }
      };
    },
    computed: {
      current() { return activePath; }
    },
    methods: {
      isActive(href) {
        // Basic active detection
        return this.current.endsWith(href) || this.current === href;
      },
      toggleTheme() {
        const html = document.documentElement;
        const theme = html.getAttribute('data-theme') || 'light';
        html.setAttribute('data-theme', theme === 'light' ? 'dark' : 'light');
      }
    },
    template: `
      <div class="sidebar">
        <div class="sidebar-header">
          <div class="avatar"></div>
          <div class="user">
            <div class="user-name">{{ user.name }}</div>
            <div class="user-email">{{ user.email }}</div>
          </div>
        </div>
        <nav class="sidebar-nav">
          <a v-for="l in links" :key="l.href" :href="l.href"
             class="nav-item" :class="{ active: isActive(l.href) }">
            <span class="material-symbols-outlined nav-icon">{{ l.icon }}</span>
            <span class="nav-label">{{ l.label }}</span>
          </a>
        </nav>
        <div class="sidebar-footer">
          <button class="btn" @click="toggleTheme">Toggle Theme</button>
        </div>
      </div>
    `
  });
  App.mount('#sidebar');
};

// Minimal styles rely on CSS variables defined in static/theme.css
// The component HTML classes are styled by those variables to keep consistency.