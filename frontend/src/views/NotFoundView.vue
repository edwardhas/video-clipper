<script setup lang="ts">
import { RouterLink } from 'vue-router'
</script>

<template>
  <div class="page page-narrow">
    <div class="panel stack">
      <h1>Page not found</h1>
      <p class="muted">This address does not match any page in Local Clipper.</p>
      <p class="row">
        <RouterLink to="/" class="btn btn-primary">Start a new project</RouterLink>
        <RouterLink to="/projects" class="btn">See all projects</RouterLink>
      </p>
    </div>
  </div>
</template>
