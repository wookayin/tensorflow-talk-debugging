'use strict';

var gulp = require('gulp');
var $ = require('gulp-load-plugins')();
var browserSync = require('browser-sync');

gulp.task('serve', function () {
  browserSync({
    notify: false,
    port: 9000,
    server: {
      baseDir: ['.']
    },
    ui: false
  });
  gulp.watch([
    'index.html',
    'style.css',
    'slides/*',
    'css/*'
  ])
  .on('change', browserSync.reload);
});

gulp.task('deploy', function () {
  return gulp.src([
    'index.html',
    'slides/*',
    'css/*',
    'node_modules/remark/**',
    'hotfix/**'
  ], {
    base: '.'
  })
  .pipe($.ghPages());
});

gulp.task('default', ['serve']);
