<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import ConfirmHost from './components/ConfirmHost.vue'
import ToastRegion from './components/ToastRegion.vue'
import { loadSession, useSession } from './stores/session'

const route = useRoute()
const { session, sessionError } = useSession()
const main = ref<HTMLElement | null>(null)

const isLogin = computed(() => route.name === 'login')
const showNav = computed(() => !isLogin.value)
const offline = computed(() => !session.value && !!sessionError.value)
const retrying = ref(false)

const NAV = [
  { to: '/', label: 'New project', match: (name: unknown) => name === 'import' },
  { to: '/projects', label: 'Projects', match: (name: unknown) => name === 'projects' || name === 'project' || name === 'clip' },
  { to: '/settings', label: 'Settings', match: (name: unknown) => name === 'settings' },
]

// Move focus to the page content after client-side navigation so screen readers announce the new page.
let firstNavigation = true
watch(
  () => route.path,
  async () => {
    if (firstNavigation) {
      firstNavigation = false
      return
    }
    await nextTick()
    main.value?.focus({ preventScroll: true })
  },
)

async function retryConnection() {
  retrying.value = true
  await loadSession(true)
  retrying.value = false
  if (session.value) window.location.reload()
}
</script>

<template>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="app-header on-dark">
    <div class="app-header-inner">
      <RouterLink to="/" class="brand" aria-label="Local Clipper, start a new project">
        <svg class="brand-mark" viewBox="0 0 32 32" aria-hidden="true" focusable="false">
          <rect width="32" height="32" rx="7" fill="#4a183d" />
          <g fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round">
            <circle cx="9.5" cy="22.5" r="3.6" />
            <circle cx="22.5" cy="22.5" r="3.6" />
            <path d="M12.2 20 23 6.5M19.8 20 9 6.5" />
          </g>
          <circle cx="16" cy="15.6" r="1.6" fill="#f2a33a" />
        </svg>
        <span class="brand-name">Local Clipper</span>
      </RouterLink>
      <nav v-if="showNav" class="app-nav" aria-label="Main">
        <RouterLink
          v-for="item in NAV"
          :key="item.to"
          :to="item.to"
          class="nav-link"
          :class="{ active: item.match(route.name) }"
          :aria-current="item.match(route.name) ? 'page' : undefined"
        >
          {{ item.label }}
        </RouterLink>
      </nav>
    </div>
  </header>

  <div v-if="offline" class="offline-bar" role="alert">
    <p>
      <strong>Cannot reach Local Clipper.</strong>
      Check that the app is still running (start it again with the start script if you closed it), then retry.
    </p>
    <button type="button" class="btn btn-sm" :aria-busy="retrying ? 'true' : undefined" @click="retryConnection">
      <span v-if="retrying" class="spinner" aria-hidden="true"></span>
      Retry
    </button>
  </div>

  <main id="main" ref="main" tabindex="-1" class="app-main">
    <RouterView v-slot="{ Component }">
      <component :is="Component" :key="route.path" />
    </RouterView>
  </main>

  <ToastRegion />
  <ConfirmHost />
</template>

<style scoped>
.skip-link {
  position: absolute;
  left: var(--s3);
  top: -100px;
  z-index: 100;
  padding: var(--s2) var(--s4);
  border-radius: var(--r-control);
  background: var(--amber);
  color: var(--brand-900);
  font-weight: 700;
  text-decoration: none;
}

.skip-link:focus {
  top: var(--s3);
}

.app-header {
  position: sticky;
  top: 0;
  z-index: 20;
  background: var(--brand);
  color: var(--white);
}

.app-header-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--s4);
  max-width: var(--content-max);
  min-height: var(--header-h);
  margin: 0 auto;
  padding: 0 var(--s5);
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: var(--s3);
  color: var(--white);
  text-decoration: none;
  border-radius: var(--r-control);
}

.brand:hover {
  color: var(--white);
}

.brand-mark {
  width: 30px;
  height: 30px;
  flex: none;
}

.brand-name {
  font-size: var(--fs-lg);
  font-weight: 700;
  letter-spacing: -0.01em;
}

.app-nav {
  display: flex;
  align-items: stretch;
  gap: var(--s1);
  align-self: stretch;
}

.nav-link {
  position: relative;
  display: inline-flex;
  align-items: center;
  padding: 0 var(--s3);
  color: #f1e3ec;
  font-size: var(--fs-sm);
  font-weight: 650;
  text-decoration: none;
  border-radius: var(--r-control);
}

.nav-link:hover {
  color: var(--white);
  background: rgba(255, 255, 255, 0.08);
}

.nav-link.active {
  color: var(--white);
}

.nav-link.active::after {
  content: '';
  position: absolute;
  left: var(--s3);
  right: var(--s3);
  bottom: 0;
  height: 3px;
  border-radius: 3px 3px 0 0;
  background: var(--amber);
}

.offline-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: var(--s3);
  padding: var(--s3) var(--s5);
  background: var(--danger-bg);
  border-bottom: 1px solid #f0c5c7;
  color: var(--ink);
  font-size: var(--fs-sm);
}

.offline-bar strong {
  color: var(--danger);
}

.app-main {
  display: block;
  min-height: calc(100vh - var(--header-h));
}

.app-main:focus-visible {
  outline: none;
  box-shadow: none;
}

@media (max-width: 640px) {
  .app-header {
    position: static;
  }

  .app-header-inner {
    flex-direction: column;
    align-items: stretch;
    gap: 0;
    padding: var(--s2) var(--s4) 0;
  }

  .brand {
    align-self: flex-start;
    padding: var(--s1) 0;
  }

  .app-nav {
    justify-content: space-between;
    margin: 0 calc(-1 * var(--s2));
  }

  .nav-link {
    flex: 1;
    justify-content: center;
    min-height: 44px;
    padding: 0 var(--s2);
  }

  .nav-link.active::after {
    left: var(--s2);
    right: var(--s2);
  }

  .offline-bar {
    padding: var(--s3) var(--s4);
  }
}
</style>
