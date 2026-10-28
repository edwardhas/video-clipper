<script setup lang="ts">
import { onMounted, ref, useId } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ApiError, api } from '../api/client'
import ActionButton from '../components/ActionButton.vue'
import AppIcon from '../components/AppIcon.vue'
import { safeNext } from '../lib/nav'

const route = useRoute()
const router = useRouter()
const uid = useId()

const username = ref('')
const password = ref('')
const busy = ref(false)
const error = ref<string | null>(null)
const userInput = ref<HTMLInputElement | null>(null)

async function submit() {
  if (busy.value) return
  if (!username.value.trim() || !password.value) {
    error.value = 'Enter your username and password.'
    return
  }
  busy.value = true
  error.value = null
  try {
    const session = await api.login({ username: username.value.trim(), password: password.value })
    if (!session.authenticated) {
      error.value = 'Signing in did not work. Try again.'
      return
    }
    password.value = ''
    await router.replace(safeNext(route.query.next))
  } catch (err) {
    error.value = err instanceof ApiError ? err.message : 'Signing in did not work. Try again.'
  } finally {
    busy.value = false
  }
}

onMounted(() => userInput.value?.focus())
</script>

<template>
  <div class="page login-page">
    <form class="panel login-card" novalidate @submit.prevent="submit">
      <h1>Sign in</h1>
      <p class="muted small">This Local Clipper is protected. Sign in with the owner account to continue.</p>

      <div class="field">
        <label class="field-label" :for="`${uid}-user`">Username</label>
        <input
          :id="`${uid}-user`"
          ref="userInput"
          v-model="username"
          class="input"
          type="text"
          name="username"
          autocomplete="username"
          autocapitalize="none"
          spellcheck="false"
          maxlength="100"
          required
        />
      </div>
      <div class="field">
        <label class="field-label" :for="`${uid}-pass`">Password</label>
        <input
          :id="`${uid}-pass`"
          v-model="password"
          class="input"
          type="password"
          name="password"
          autocomplete="current-password"
          maxlength="1000"
          required
        />
      </div>

      <div v-if="error" class="notice notice-error" role="alert">
        <AppIcon name="alert" />
        <p>{{ error }}</p>
      </div>

      <ActionButton type="submit" variant="primary" size="lg" block :loading="busy" loading-label="Signing in…">
        Sign in
      </ActionButton>
    </form>
  </div>
</template>

<style scoped>
.login-page {
  display: grid;
  place-items: start center;
  padding-top: var(--s8);
}

.login-card {
  width: 100%;
  max-width: 400px;
  display: flex;
  flex-direction: column;
  gap: var(--s4);
  border-top: 4px solid var(--brand);
}

.login-card h1 {
  font-size: var(--fs-xl);
}

@media (max-width: 640px) {
  .login-page {
    padding-top: var(--s6);
  }
}
</style>
