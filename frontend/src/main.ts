import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import { setUnauthorizedHandler } from './api/client'
import { router } from './router'
import { markSignedOut } from './stores/session'

// Any 401 from the API means the session ended: go to the sign-in page and come back afterwards.
setUnauthorizedHandler(() => {
  markSignedOut()
  const current = router.currentRoute.value
  if (current.name === 'login') return
  void router.replace({ name: 'login', query: { next: current.fullPath } })
})

createApp(App).use(router).mount('#app')
