import { createRouter, createWebHistory } from 'vue-router'
import { safeNext } from './lib/nav'
import { loadSession } from './stores/session'

declare module 'vue-router' {
  interface RouteMeta {
    title?: string
    /** Routes reachable without a session (only the login page). */
    public?: boolean
  }
}

export const router = createRouter({
  history: createWebHistory('/'),
  routes: [
    { path: '/', name: 'import', component: () => import('./views/ImportView.vue'), meta: { title: 'New project' } },
    {
      path: '/projects',
      name: 'projects',
      component: () => import('./views/ProjectsView.vue'),
      meta: { title: 'Projects' },
    },
    {
      path: '/projects/:id',
      name: 'project',
      component: () => import('./views/ProjectView.vue'),
      props: true,
      meta: { title: 'Project' },
    },
    {
      path: '/clips/:id',
      name: 'clip',
      component: () => import('./views/ClipEditorView.vue'),
      props: true,
      meta: { title: 'Edit clip' },
    },
    {
      path: '/settings',
      name: 'settings',
      component: () => import('./views/SettingsView.vue'),
      meta: { title: 'Settings' },
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('./views/LoginView.vue'),
      meta: { title: 'Sign in', public: true },
    },
    {
      path: '/:pathMatch(.*)*',
      name: 'not-found',
      component: () => import('./views/NotFoundView.vue'),
      meta: { title: 'Page not found' },
    },
  ],
  scrollBehavior(to, from, saved) {
    if (saved) return saved
    if (to.path === from.path) return false
    return { top: 0 }
  },
})

router.beforeEach(async (to) => {
  const session = await loadSession()
  // Backend unreachable: let the page render and show its own connection error.
  if (!session) return true
  const needsLogin = session.auth_required && !session.authenticated
  if (to.name === 'login') {
    return needsLogin ? true : safeNext(to.query.next)
  }
  if (needsLogin) return { name: 'login', query: { next: to.fullPath } }
  return true
})

router.afterEach((to) => {
  document.title = to.meta.title ? `${to.meta.title} – Local Clipper` : 'Local Clipper'
})
